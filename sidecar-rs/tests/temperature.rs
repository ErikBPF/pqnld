use std::process::{Command, Stdio};

#[test]
fn cli_temperature_requires_finite_positive_value() {
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    let directory = target.join(format!("temperature-fixture-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("fixture.json"), r#"{"readout":"echo"}"#).unwrap();
    for source in ["cli", "PQNLD_TEMPERATURE", "DECISION_TEMPERATURE", "descriptor"] {
        for (value, accepted) in [("NaN", false), ("inf", false), ("-inf", false),
            ("0", false), ("-1", false), ("1", true), ("0.5", true), ("1e-9", true)] {
            if source == "descriptor" {
                // JSON cannot represent NaN/infinity; valid finite descriptor cases only.
                if matches!(value, "NaN" | "inf" | "-inf") { continue; }
                std::fs::write(directory.join("fixture.json"), format!(r#"{{"readout":"echo","temperature":{value}}}"#)).unwrap();
            }
            let mut command = Command::new(env!("CARGO_BIN_EXE_pqnld-rs"));
            command.args(["--mcp", "--engine-kind", "vllm", "--vllm-url", "http://127.0.0.1:1",
                "--descriptor", "fixture", "--models-dir"])
                .arg(&directory).stdin(Stdio::null())
                .env_remove("PQNLD_TEMPERATURE").env_remove("DECISION_TEMPERATURE");
            if source == "cli" { command.args(["--temperature", value]); }
            else if source != "descriptor" { command.env(source, value); }
            // Explicit echo descriptor + MCP EOF: startup never contacts an engine.
            let output = command.output().unwrap();
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert_eq!(output.status.success(), accepted, "{source}={value}: {stderr}");
            if accepted { assert!(stderr.contains("readout=echo"), "{stderr}"); }
            else { assert!(stderr.contains("temperature"), "{stderr}"); }
        }
    }
    std::fs::remove_file(directory.join("fixture.json")).unwrap();
    std::fs::remove_dir(directory).unwrap();
}
