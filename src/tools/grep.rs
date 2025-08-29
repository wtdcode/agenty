use std::path::PathBuf;

use grep::{
    printer::StandardBuilder,
    regex::RegexMatcher,
    searcher::{BinaryDetection, SearcherBuilder},
};
use log::warn;
use schemars::JsonSchema;
use serde::Deserialize;
use walkdir::WalkDir;

use crate::{error::AgentyError, tool::Tool};

use super::file::sanitize_join_relative_path;

#[derive(JsonSchema, Deserialize)]
pub struct GrepToolArgs {
    pub directory: PathBuf,
    pub pattern: String,
}

#[derive(Debug, Clone)]
pub struct GrepTool {
    pub cwd: PathBuf,
}

impl GrepTool {
    pub fn new(cwd: PathBuf) -> Self {
        Self { cwd }
    }
    pub async fn grep(&self, directory: PathBuf, pattern: String) -> Result<String, AgentyError> {
        let target_path = match sanitize_join_relative_path(&self.cwd, &directory) {
            Ok(p) => p,
            Err(e) => return Ok(e),
        };
        if !target_path.is_dir() {
            return Ok(format!("{:?} is not a directory", &directory));
        }

        tokio::task::spawn_blocking(move || {
            let mut buf = vec![];
            let cur = std::io::Cursor::new(&mut buf);
            let mut printer = StandardBuilder::new()
                .column(true)
                .max_columns(Some(80))
                .build_no_color(cur);
            let mut searcher = SearcherBuilder::new()
                .binary_detection(BinaryDetection::quit(b'\x00'))
                .line_number(true)
                .build();
            let matcher = match RegexMatcher::new_line_matcher(&pattern) {
                Ok(v) => v,
                Err(e) => return Ok(format!("regex {} error with {}", pattern, e)),
            };

            for result in WalkDir::new(&target_path) {
                let dent = match result {
                    Ok(dent) => dent,
                    Err(err) => {
                        warn!("Fail to walk due to {}", err);
                        continue;
                    }
                };
                if !dent.file_type().is_file() {
                    continue;
                }
                if let Err(e) = searcher.search_path(
                    &matcher,
                    dent.path(),
                    printer.sink_with_path(&matcher, dent.path()),
                ) {
                    warn!("Fail to search {:?} due to {}", &dent, e);
                }
            }
            let mut resp = String::from_utf8_lossy(&buf).to_string();
            if resp.len() > 16384 {
                // cutoff a bit...
                resp = (&resp[0..16384]).to_string();
            }
            Ok(resp)
        })
        .await?
    }
}

impl Tool for GrepTool {
    type ARGUMENTS = GrepToolArgs;
    const NAME: &str = "grep_files";
    const DESCRIPTION: Option<&str> = Some(
        "Grep files in the given path with pattern. The path should be always relative path and '.' is allowed while '..' is not allowed. Note the pattern is in regex grammar not glob grammar.",
    );

    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, AgentyError>> + Send {
        self.grep(arguments.directory, arguments.pattern)
    }
}
