use thiserror::Error;

macro_rules! trivial {
    ($err:ty, $var:expr) => {
        impl From<$err> for AgentyError {
            fn from(value: $err) -> Self {
                $var(value.into())
            }
        }
    };
}

macro_rules! trivial_other {
    ($err:ty) => {
        trivial!($err, AgentyError::Other);
    };
}

#[derive(Error, Debug)]
pub enum AgentyError {
    #[error("incorrect tool call, schema: {0:?}, args: {1}")]
    IncorrectToolCall(schemars::Schema, String),
    #[error("No such tool")]
    NoSuchTool(String),
    #[error("unexpected llm response: {0}")]
    Unexpected(String),
    #[error("json error: {0}")]
    STDJSON(#[from] serde_json::Error),
    #[error("prompt: {0}")]
    Prompt(#[from] openai_models::error::PromptError),
    #[error("reqwest: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("io error: {0}")]
    IO(#[from] std::io::Error),
    #[error("driver: {0}")]
    WebDriver(#[from] thirtyfour::error::WebDriverError),
    #[error("glob: {0}")]
    Glob(#[from] glob::PatternError),
    #[error("smtlib parse error: {0}")]
    SMTPARSE(String),
    #[error("z3 expression error: {0}")]
    Z3EXPR(String),
    #[error(transparent)]
    Other(color_eyre::Report),
}

trivial!(
    openai_models::openai::error::OpenAIError,
    AgentyError::Prompt
);
trivial_other!(color_eyre::Report);
trivial_other!(walkdir::Error);
trivial_other!(tokio::task::JoinError);
