use std::time::Duration;

use crate::{
    error::AgentyError,
    tool::{Tool, ToolBox},
};
use color_eyre::eyre::eyre;
use itertools::Itertools;
use log::{debug, warn};
use openai_models::llm::{LLM, LLMSettings};
use openai_models::openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    CreateChatCompletionResponse, FinishReason,
};

pub struct Agent {
    pub tools: ToolBox,
    pub system: String,
    pub user: String,
    pub context: Vec<ChatCompletionRequestMessage>,
}

#[derive(Debug, Clone)]
pub enum AgentAction<T = ()> {
    Continue,
    Unexpected(String),
    Out(T),
}

impl Agent {
    fn full_context(&self) -> Vec<ChatCompletionRequestMessage> {
        vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(self.system.clone())
                    .build()
                    .unwrap(),
            ),
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(self.user.clone())
                    .build()
                    .unwrap(),
            ),
        ]
        .into_iter()
        .chain(self.context.clone().into_iter())
        .collect()
    }

    pub fn new(tools: ToolBox, system: Option<String>, user: String) -> Self {
        let system = system.unwrap_or(
            "You are an expert agent that calls tool to complete your task.".to_string(),
        );
        Self {
            tools,
            system,
            user,
            context: vec![],
        }
    }

    pub async fn run_once<TC, MS, RF, T>(
        &mut self,
        llm: &mut LLM,
        prefix: Option<&str>,
        settings: Option<LLMSettings>,
        on_toolcalls: TC,
        on_message: MS,
        on_refusal: RF,
    ) -> Result<AgentAction<T>, AgentyError>
    where
        TC: AsyncFnOnce(
            &mut Self,
            Vec<ChatCompletionMessageToolCall>,
        ) -> Result<AgentAction<T>, AgentyError>,
        MS: AsyncFnOnce(&mut Self, String) -> Result<AgentAction<T>, AgentyError>,
        RF: AsyncFnOnce(&mut Self, String) -> Result<AgentAction<T>, AgentyError>,
    {
        let settings = settings.unwrap_or_else(|| llm.default_settings.clone());
        let req = CreateChatCompletionRequestArgs::default()
            .tools(self.tools.openai_objects())
            .messages(self.full_context())
            .model(llm.model.to_string())
            .temperature(settings.llm_temperature)
            .presence_penalty(settings.llm_presence_penalty)
            .max_completion_tokens(settings.llm_max_completion_tokens)
            .tool_choice(settings.llm_tool_choice)
            .build()?;
        let timeout = Duration::from_secs(settings.llm_prompt_timeout);

        let mut resp: CreateChatCompletionResponse = llm
            .complete_once_with_retry(&req, prefix, Some(timeout), Some(settings.llm_retry))
            .await?;

        let choice = resp.choices.swap_remove(0);

        if matches!(choice.finish_reason, Some(FinishReason::ToolCalls))
            || choice
                .message
                .tool_calls
                .as_ref()
                .map(|t| !t.is_empty())
                .unwrap_or_default()
        {
            self.context.push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(choice.message.tool_calls.clone().unwrap_or_default())
                    .build()?,
            ));
            on_toolcalls(self, choice.message.tool_calls.unwrap_or_default()).await
        } else if matches!(choice.finish_reason, Some(FinishReason::ContentFilter))
            || choice.message.refusal.is_some()
        {
            self.context.push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .refusal(choice.message.refusal.clone().unwrap_or_default())
                    .build()?,
            ));
            on_refusal(self, choice.message.refusal.unwrap_or_default()).await
        } else if matches!(choice.finish_reason, Some(FinishReason::Stop))
            || matches!(choice.finish_reason, Some(FinishReason::Length))
            || choice.message.content.is_some()
        {
            self.context.push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .content(choice.message.content.clone().unwrap_or_default())
                    .build()?,
            ));
            on_message(self, choice.message.content.unwrap_or_default()).await
        } else {
            Err(AgentyError::Other(eyre!(
                "Not supported choice: {:?}",
                &choice
            )))
        }
    }

    async fn handle_toolcalls(
        &mut self,
        toolcalls: Vec<ChatCompletionMessageToolCall>,
    ) -> Result<Vec<String>, AgentyError> {
        let mut resps = vec![];
        for call in toolcalls {
            match self
                .tools
                .invoke(call.function.name.clone(), call.function.arguments)
                .await
            {
                None => {
                    warn!("No such tool: {}, will try again", &call.function.name);
                    return Err(AgentyError::NoSuchTool(call.function.name));
                }
                Some(Ok(v)) => resps.push(v),
                Some(Err(e)) => return Err(e),
            }
        }
        Ok(resps)
    }

    pub fn append_user(&mut self, ctx: String) -> Result<(), AgentyError> {
        let user = ChatCompletionRequestUserMessageArgs::default()
            .content(ctx)
            .build()?;
        self.append_context(ChatCompletionRequestMessage::User(user));
        Ok(())
    }

    pub fn append_context(&mut self, ctx: ChatCompletionRequestMessage) {
        self.context.push(ctx);
    }

    pub fn revert_context(&mut self) {
        self.context.pop();
    }

    pub async fn run_until_tool<T: Tool>(
        &mut self,
        llm: &mut LLM,
        prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<T::ARGUMENTS, AgentyError> {
        loop {
            let action = self
                .run_once(
                    llm,
                    prefix,
                    settings.clone(),
                    async |ctx, toolcalls| {
                        if let Some(call) = toolcalls.iter().find(|t| t.function.name == T::NAME) {
                            let td: T::ARGUMENTS = serde_json::from_str(&call.function.arguments)?;
                            Ok(AgentAction::Out(td))
                        } else {
                            let resps = match ctx.handle_toolcalls(toolcalls).await {
                                Ok(v) => v,
                                Err(e) => match &e {
                                    AgentyError::NoSuchTool(_)
                                    | AgentyError::IncorrectToolCall(_, _) => {
                                        warn!("Error {} during tool call, retry...", e);
                                        return Ok(AgentAction::Continue);
                                    }
                                    _ => return Err(e),
                                },
                            };
                            ctx.append_user(resps.into_iter().join("\n"))?;
                            Ok(AgentAction::Continue)
                        }
                    },
                    async |_, msg| Ok(AgentAction::Unexpected(msg)),
                    async |_, msg| Ok(AgentAction::Unexpected(msg)),
                )
                .await?;

            match action {
                AgentAction::Continue => continue,
                AgentAction::Unexpected(s) => return Err(AgentyError::Unexpected(s)),
                AgentAction::Out(s) => return Ok(s),
            }
        }
    }

    pub async fn run_until_text(
        &mut self,
        llm: &mut LLM,
        prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<String, AgentyError> {
        loop {
            let action = self
                .run_once(
                    llm,
                    prefix,
                    settings.clone(),
                    async |ctx, toolcalls| {
                        let resps = match ctx.handle_toolcalls(toolcalls).await {
                            Ok(v) => v,
                            Err(e) => match &e {
                                AgentyError::NoSuchTool(_)
                                | AgentyError::IncorrectToolCall(_, _) => {
                                    warn!("Error {} during tool call, retry...", e);
                                    return Ok(AgentAction::Continue);
                                }
                                _ => return Err(e),
                            },
                        };
                        ctx.append_user(resps.into_iter().join("\n"))?;
                        Ok(AgentAction::Continue)
                    },
                    async |_, msg| Ok(AgentAction::Out(msg)),
                    async |_, msg| Ok(AgentAction::Unexpected(msg)),
                )
                .await?;
            debug!("Agent action: {:?}", &action);
            match action {
                AgentAction::Continue => continue,
                AgentAction::Unexpected(s) => return Ok(s),
                AgentAction::Out(s) => return Ok(s),
            }
        }
    }
}
