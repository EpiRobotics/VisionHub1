using Newtonsoft.Json;

namespace VisionHubUI.Models;

/// <summary>
/// Health check response from GET /health
/// </summary>
public class HealthInfo
{
    [JsonProperty("ok")]
    public bool Ok { get; set; }

    [JsonProperty("service")]
    public string Service { get; set; } = "";

    [JsonProperty("version")]
    public string Version { get; set; } = "";

    [JsonProperty("uptime_s")]
    public double UptimeSeconds { get; set; }

    [JsonProperty("projects_count")]
    public int ProjectsCount { get; set; }

    [JsonProperty("gpu_workers")]
    public int GpuWorkers { get; set; }

    [JsonProperty("timestamp")]
    public string Timestamp { get; set; } = "";
}
