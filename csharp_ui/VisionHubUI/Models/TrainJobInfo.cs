using Newtonsoft.Json;

namespace VisionHubUI.Models;

/// <summary>
/// Training job status from GET /train/{train_job_id}
/// </summary>
public class TrainJobInfo
{
    [JsonProperty("train_job_id")]
    public string TrainJobId { get; set; } = "";

    [JsonProperty("project_id")]
    public string ProjectId { get; set; } = "";

    [JsonProperty("status")]
    public string Status { get; set; } = "";

    [JsonProperty("progress")]
    public double Progress { get; set; }

    [JsonProperty("message")]
    public string Message { get; set; } = "";

    [JsonProperty("started_at")]
    public string StartedAt { get; set; } = "";

    [JsonProperty("completed_at")]
    public string CompletedAt { get; set; } = "";

    [JsonProperty("model_version")]
    public string ModelVersion { get; set; } = "";

    [JsonProperty("out_model_dir")]
    public string OutModelDir { get; set; } = "";

    [JsonProperty("log_lines")]
    public List<string> LogLines { get; set; } = new();

    [JsonProperty("error")]
    public string? Error { get; set; }
}

/// <summary>
/// Response from POST /projects/{id}/train
/// </summary>
public class TrainStartResponse
{
    [JsonProperty("ok")]
    public bool Ok { get; set; }

    [JsonProperty("train_job_id")]
    public string TrainJobId { get; set; } = "";

    [JsonProperty("project_id")]
    public string ProjectId { get; set; } = "";
}
