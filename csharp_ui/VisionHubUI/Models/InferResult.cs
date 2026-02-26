using Newtonsoft.Json;

namespace VisionHubUI.Models;

/// <summary>
/// Unified inference result schema.
/// </summary>
public class InferResult
{
    [JsonProperty("job_id")]
    public string JobId { get; set; } = "";

    [JsonProperty("project_id")]
    public string ProjectId { get; set; } = "";

    [JsonProperty("ok")]
    public bool Ok { get; set; }

    [JsonProperty("pred")]
    public string Pred { get; set; } = "";

    [JsonProperty("score")]
    public double Score { get; set; }

    [JsonProperty("threshold")]
    public double Threshold { get; set; }

    [JsonProperty("timing_ms")]
    public TimingInfo? TimingMs { get; set; }

    [JsonProperty("artifacts")]
    public ArtifactsInfo? Artifacts { get; set; }

    [JsonProperty("regions")]
    public List<RegionInfo>? Regions { get; set; }

    [JsonProperty("model_version")]
    public string ModelVersion { get; set; } = "";

    [JsonProperty("error")]
    public ErrorInfo? Error { get; set; }
}

public class TimingInfo
{
    [JsonProperty("wait_file")]
    public double WaitFile { get; set; }

    [JsonProperty("read")]
    public double Read { get; set; }

    [JsonProperty("infer")]
    public double Infer { get; set; }

    [JsonProperty("post")]
    public double Post { get; set; }

    [JsonProperty("save")]
    public double Save { get; set; }

    [JsonProperty("total")]
    public double Total { get; set; }
}

public class ArtifactsInfo
{
    [JsonProperty("u16")]
    public string U16 { get; set; } = "";

    [JsonProperty("mask")]
    public string Mask { get; set; } = "";

    [JsonProperty("heatmap")]
    public string Heatmap { get; set; } = "";

    [JsonProperty("overlay")]
    public string Overlay { get; set; } = "";
}

public class RegionInfo
{
    [JsonProperty("x")]
    public int X { get; set; }

    [JsonProperty("y")]
    public int Y { get; set; }

    [JsonProperty("w")]
    public int W { get; set; }

    [JsonProperty("h")]
    public int H { get; set; }

    [JsonProperty("score")]
    public double Score { get; set; }

    [JsonProperty("area_px")]
    public int AreaPx { get; set; }
}

public class ErrorInfo
{
    [JsonProperty("code")]
    public string Code { get; set; } = "";

    [JsonProperty("message")]
    public string Message { get; set; } = "";
}
