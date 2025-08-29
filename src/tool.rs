use std::future::Future;
use std::pin::Pin;
use std::{collections::HashMap, fmt::Debug};

use dyn_clone::DynClone;
use log::debug;
use openai_models::openai::types::{ChatCompletionTool, ChatCompletionToolType, FunctionObject};
use schemars::schema_for;
use serde::de::DeserializeOwned;

use crate::error::AgentyError;

pub trait ToolDyn: DynClone + Debug + std::any::Any {
    fn name(&self) -> String;
    fn to_openai_obejct(&self) -> ChatCompletionTool;
    fn call(
        &self,
        arguments: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, AgentyError>> + Send + '_>>;
}

pub fn downcast_tool<T: 'static>(tool: &dyn ToolDyn) -> &T {
    (tool as &dyn std::any::Any)
        .downcast_ref::<T>()
        .expect("can not downcast")
}

dyn_clone::clone_trait_object!(ToolDyn);

pub trait Tool: Send + Sync + DynClone + Debug {
    type ARGUMENTS: DeserializeOwned + schemars::JsonSchema + Sized + Send;
    const NAME: &str;
    const DESCRIPTION: Option<&str>;
    const STRICT: bool = false;

    fn to_openai_obejct(&self) -> ChatCompletionTool {
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: Self::NAME.to_string(),
                description: Self::DESCRIPTION.map(|e| e.to_string()),
                parameters: Some(
                    serde_json::to_value(schema_for!(Self::ARGUMENTS))
                        .expect("Fail to generate schema?!"),
                ),
                strict: Some(Self::STRICT),
            },
        }
    }
    fn call(&self, arguments: String) -> impl Future<Output = Result<String, AgentyError>> + Send {
        async move {
            match serde_json::from_str::<Self::ARGUMENTS>(&arguments) {
                Ok(args) => self.invoke(args).await,
                Err(_) => Err(AgentyError::IncorrectToolCall(
                    schema_for!(Self::ARGUMENTS),
                    arguments,
                )),
            }
        }
    }

    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, AgentyError>> + Send;
}

impl<T: Tool + DynClone + 'static> ToolDyn for T {
    fn name(&self) -> String {
        Self::NAME.to_string()
    }
    fn call(
        &self,
        arguments: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, AgentyError>> + Send + '_>> {
        Box::pin(self.call(arguments))
    }

    fn to_openai_obejct(&self) -> ChatCompletionTool {
        self.to_openai_obejct()
    }
}

#[derive(Default, Clone, Debug)]
pub struct ToolBox {
    pub tools: HashMap<String, Box<dyn ToolDyn>>,
}

impl ToolBox {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn openai_objects(&self) -> Vec<ChatCompletionTool> {
        self.tools.iter().map(|t| t.1.to_openai_obejct()).collect()
    }

    pub fn add_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.add_dyn_tool(Box::new(tool) as _);
    }

    pub fn add_dyn_tool(&mut self, tool: Box<dyn ToolDyn>) {
        self.tools.insert(tool.name(), tool);
    }

    pub async fn invoke(
        &self,
        tool_name: String,
        arguments: String,
    ) -> Option<Result<String, AgentyError>> {
        if let Some(tool) = self.tools.get(&tool_name) {
            debug!("Invoking tool {} with arguments {}", &tool_name, &arguments);
            Some(tool.call(arguments).await)
        } else {
            None
        }
    }
}
