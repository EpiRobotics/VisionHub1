using Newtonsoft.Json;

namespace VisionHubUI.Models;

/// <summary>
/// Response from POST /label/crop
/// </summary>
public class LabelCropResponse
{
    [JsonProperty("ok")]
    public bool Ok { get; set; }

    [JsonProperty("total_crops")]
    public int TotalCrops { get; set; }

    [JsonProperty("processed_files")]
    public int ProcessedFiles { get; set; }

    [JsonProperty("total_json_files")]
    public int TotalJsonFiles { get; set; }

    [JsonProperty("classes")]
    public List<LabelClassInfo> Classes { get; set; } = new();

    [JsonProperty("errors")]
    public List<string> Errors { get; set; } = new();
}

/// <summary>
/// Class info returned from crop/scan operations
/// </summary>
public class LabelClassInfo
{
    [JsonProperty("class")]
    public string ClassName { get; set; } = "";

    [JsonProperty("count")]
    public int Count { get; set; }
}

/// <summary>
/// Response from POST /label/scan_bank
/// </summary>
public class LabelScanBankResponse
{
    [JsonProperty("ok")]
    public bool Ok { get; set; }

    [JsonProperty("bank_dir")]
    public string BankDir { get; set; } = "";

    [JsonProperty("total_classes")]
    public int TotalClasses { get; set; }

    [JsonProperty("total_images")]
    public int TotalImages { get; set; }

    [JsonProperty("classes")]
    public List<LabelClassInfo> Classes { get; set; } = new();
}

/// <summary>
/// Response from POST /label/train (start training)
/// </summary>
public class LabelTrainStartResponse
{
    [JsonProperty("ok")]
    public bool Ok { get; set; }

    [JsonProperty("job_id")]
    public string JobId { get; set; } = "";
}

/// <summary>
/// Response from GET /label/train/{job_id} (training status)
/// </summary>
public class LabelTrainStatus
{
    [JsonProperty("job_id")]
    public string JobId { get; set; } = "";

    [JsonProperty("status")]
    public string Status { get; set; } = "";

    [JsonProperty("progress")]
    public double Progress { get; set; }

    [JsonProperty("message")]
    public string Message { get; set; } = "";

    [JsonProperty("log_lines")]
    public List<string> LogLines { get; set; } = new();

    [JsonProperty("result")]
    public object? Result { get; set; }

    [JsonProperty("error")]
    public string? Error { get; set; }

    [JsonProperty("started_at")]
    public string StartedAt { get; set; } = "";

    [JsonProperty("completed_at")]
    public string CompletedAt { get; set; } = "";
}
