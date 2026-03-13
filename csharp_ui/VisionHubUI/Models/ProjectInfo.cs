using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace VisionHubUI.Models;

/// <summary>
/// Project summary returned by GET /projects
/// </summary>
public class ProjectInfo
{
    [JsonProperty("project_id")]
    public string ProjectId { get; set; } = "";

    [JsonProperty("display_name")]
    public string DisplayName { get; set; } = "";

    [JsonProperty("enabled")]
    public bool Enabled { get; set; }

    [JsonProperty("tcp_port")]
    public int TcpPort { get; set; }

    [JsonProperty("algo")]
    public string Algo { get; set; } = "";

    [JsonProperty("model_loaded")]
    public bool ModelLoaded { get; set; }

    [JsonProperty("model_version")]
    public string ModelVersion { get; set; } = "";

    [JsonProperty("tcp_running")]
    public bool TcpRunning { get; set; }

    [JsonProperty("stats")]
    public ProjectStats? Stats { get; set; }

    /// <summary>
    /// Full nested config from GET /projects/{id} detail endpoint.
    /// Not present in list endpoint.
    /// </summary>
    [JsonProperty("config")]
    public JObject? Config { get; set; }

    public override string ToString()
    {
        string status = Enabled ? "ON" : "OFF";
        return $"[{status}] {ProjectId} - {DisplayName} (:{TcpPort})";
    }
}

/// <summary>
/// Runtime statistics for a project.
/// </summary>
public class ProjectStats
{
    [JsonProperty("total_jobs")]
    public int TotalJobs { get; set; }

    [JsonProperty("ok_count")]
    public int OkCount { get; set; }

    [JsonProperty("ng_count")]
    public int NgCount { get; set; }

    [JsonProperty("error_count")]
    public int ErrorCount { get; set; }

    [JsonProperty("avg_infer_ms")]
    public double AvgInferMs { get; set; }

    [JsonProperty("last_result_time")]
    public string LastResultTime { get; set; } = "";
}

/// <summary>
/// Log entry from GET /projects/{id}/logs
/// </summary>
public class LogEntry
{
    [JsonProperty("ts")]
    public string Timestamp { get; set; } = "";

    [JsonProperty("level")]
    public string Level { get; set; } = "";

    [JsonProperty("source")]
    public string Source { get; set; } = "";

    [JsonProperty("msg")]
    public string Message { get; set; } = "";
}

/// <summary>
/// Response from GET /projects/{id}/logs
/// </summary>
public class ProjectLogsResponse
{
    [JsonProperty("entries")]
    public List<LogEntry> Entries { get; set; } = new();

    [JsonProperty("next_index")]
    public int NextIndex { get; set; }
}
