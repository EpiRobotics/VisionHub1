using System.Net.Http.Headers;
using System.Text;
using Newtonsoft.Json;
using VisionHubUI.Models;

namespace VisionHubUI.Services;

/// <summary>
/// HTTP client for communicating with the VisionHub Python AI Service.
/// All UI-to-service communication goes through this client.
/// </summary>
public class VisionHubApiClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private string _baseUrl;

    public VisionHubApiClient(string baseUrl = "http://localhost:8100")
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(30)
        };
        _httpClient.DefaultRequestHeaders.Accept.Add(
            new MediaTypeWithQualityHeaderValue("application/json"));
    }

    public string BaseUrl
    {
        get => _baseUrl;
        set => _baseUrl = value.TrimEnd('/');
    }

    // ------------------------------------------------------------------
    // Health
    // ------------------------------------------------------------------

    public async Task<HealthInfo?> GetHealthAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/health");
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<HealthInfo>(json);
        }
        catch (Exception)
        {
            return null;
        }
    }

    // ------------------------------------------------------------------
    // Projects
    // ------------------------------------------------------------------

    public async Task<List<ProjectInfo>> GetProjectsAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/projects");
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<List<ProjectInfo>>(json) ?? new List<ProjectInfo>();
        }
        catch (Exception)
        {
            return new List<ProjectInfo>();
        }
    }

    public async Task<ProjectInfo?> GetProjectAsync(string projectId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/projects/{projectId}");
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<ProjectInfo>(json);
        }
        catch (Exception)
        {
            return null;
        }
    }

    public async Task<bool> ReloadAllProjectsAsync()
    {
        try
        {
            var response = await _httpClient.PostAsync($"{_baseUrl}/projects/reload", null);
            return response.IsSuccessStatusCode;
        }
        catch (Exception)
        {
            return false;
        }
    }

    public async Task<bool> EnableProjectAsync(string projectId, bool enabled)
    {
        try
        {
            var content = new StringContent(
                JsonConvert.SerializeObject(new { enabled }),
                Encoding.UTF8,
                "application/json");
            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/projects/{projectId}/enable", content);
            return response.IsSuccessStatusCode;
        }
        catch (Exception)
        {
            return false;
        }
    }

    // ------------------------------------------------------------------
    // Model management
    // ------------------------------------------------------------------

    public async Task<bool> LoadModelAsync(string projectId)
    {
        try
        {
            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/projects/{projectId}/load_model", null);
            return response.IsSuccessStatusCode;
        }
        catch (Exception)
        {
            return false;
        }
    }

    public async Task<bool> SetModelAsync(string projectId, string version)
    {
        try
        {
            var content = new StringContent(
                JsonConvert.SerializeObject(new { version }),
                Encoding.UTF8,
                "application/json");
            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/projects/{projectId}/set_model", content);
            return response.IsSuccessStatusCode;
        }
        catch (Exception)
        {
            return false;
        }
    }

    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------

    public async Task<TrainStartResponse?> StartTrainingAsync(string projectId, bool autoActivate = true)
    {
        try
        {
            var content = new StringContent(
                JsonConvert.SerializeObject(new { auto_activate = autoActivate }),
                Encoding.UTF8,
                "application/json");
            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/projects/{projectId}/train", content);
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<TrainStartResponse>(json);
        }
        catch (Exception)
        {
            return null;
        }
    }

    public async Task<TrainJobInfo?> GetTrainStatusAsync(string trainJobId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/train/{trainJobId}");
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<TrainJobInfo>(json);
        }
        catch (Exception)
        {
            return null;
        }
    }

    // ------------------------------------------------------------------
    // Inference (test from UI)
    // ------------------------------------------------------------------

    public async Task<InferResult?> RunInferenceAsync(string projectId, string imagePath, string? jobId = null)
    {
        try
        {
            var payload = new
            {
                image_path = imagePath,
                job_id = jobId ?? $"ui_test_{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}"
            };
            var content = new StringContent(
                JsonConvert.SerializeObject(payload),
                Encoding.UTF8,
                "application/json");
            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/projects/{projectId}/infer", content);
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<InferResult>(json);
        }
        catch (Exception)
        {
            return null;
        }
    }

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------

    public async Task<ProjectStats?> GetProjectStatsAsync(string projectId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/projects/{projectId}/stats");
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            var data = JsonConvert.DeserializeObject<Dictionary<string, object>>(json);
            if (data != null && data.ContainsKey("stats"))
            {
                var statsJson = JsonConvert.SerializeObject(data["stats"]);
                return JsonConvert.DeserializeObject<ProjectStats>(statsJson);
            }
            return null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    // ------------------------------------------------------------------
    // Project Logs
    // ------------------------------------------------------------------

    public async Task<ProjectLogsResponse?> GetProjectLogsAsync(string projectId, int sinceIndex = 0)
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/projects/{projectId}/logs?since={sinceIndex}");
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<ProjectLogsResponse>(json);
        }
        catch (Exception)
        {
            return null;
        }
    }

    public async Task<bool> ClearProjectLogsAsync(string projectId)
    {
        try
        {
            var response = await _httpClient.PostAsync($"{_baseUrl}/projects/{projectId}/logs/clear", null);
            return response.IsSuccessStatusCode;
        }
        catch (Exception)
        {
            return false;
        }
    }

    // ------------------------------------------------------------------
    // Project Runtime Config
    // ------------------------------------------------------------------

    public async Task<bool> SetOverlayPathAsync(string projectId, string overlayPath)
    {
        try
        {
            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/projects/{projectId}/set_overlay_path?overlay_path={Uri.EscapeDataString(overlayPath)}",
                null);
            return response.IsSuccessStatusCode;
        }
        catch (Exception)
        {
            return false;
        }
    }

    // ------------------------------------------------------------------
    // Label Training Workflow
    // ------------------------------------------------------------------

    public async Task<LabelCropResponse?> LabelCropGlyphsAsync(
        string imageDir, string jsonDir, string outputDir, int pad = 2)
    {
        try
        {
            var payload = new
            {
                image_dir = imageDir,
                json_dir = jsonDir,
                output_dir = outputDir,
                pad
            };
            var content = new StringContent(
                JsonConvert.SerializeObject(payload),
                Encoding.UTF8,
                "application/json");
            var response = await _httpClient.PostAsync($"{_baseUrl}/label/crop", content);
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<LabelCropResponse>(json);
        }
        catch (Exception)
        {
            return null;
        }
    }

    public async Task<LabelScanBankResponse?> LabelScanBankAsync(string bankDir)
    {
        try
        {
            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/label/scan_bank?bank_dir={Uri.EscapeDataString(bankDir)}", null);
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<LabelScanBankResponse>(json);
        }
        catch (Exception)
        {
            return null;
        }
    }

    public async Task<LabelTrainStartResponse?> LabelStartTrainingAsync(
        string bankDir, string outputModelDir, string projectId = "",
        bool autoActivate = true, int imgSize = 128,
        int maxPatchesPerClass = 30000, int k = 1,
        string scoreMode = "topk", int topk = 10, double pThr = 0.995)
    {
        try
        {
            var payload = new
            {
                bank_dir = bankDir,
                output_model_dir = outputModelDir,
                project_id = projectId,
                auto_activate = autoActivate,
                img_size = imgSize,
                max_patches_per_class = maxPatchesPerClass,
                k,
                score_mode = scoreMode,
                topk,
                p_thr = pThr
            };
            var content = new StringContent(
                JsonConvert.SerializeObject(payload),
                Encoding.UTF8,
                "application/json");
            // Training can be long, use extended timeout
            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(2));
            var response = await _httpClient.PostAsync($"{_baseUrl}/label/train", content, cts.Token);
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<LabelTrainStartResponse>(json);
        }
        catch (Exception)
        {
            return null;
        }
    }

    public async Task<LabelTrainStatus?> LabelGetTrainStatusAsync(string jobId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/label/train/{jobId}");
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<LabelTrainStatus>(json);
        }
        catch (Exception)
        {
            return null;
        }
    }

    // ------------------------------------------------------------------
    // TCP Port test (direct TCP ping)
    // ------------------------------------------------------------------

    public async Task<bool> PingTcpPortAsync(string host, int port, int timeoutMs = 2000)
    {
        try
        {
            using var client = new System.Net.Sockets.TcpClient();
            var connectTask = client.ConnectAsync(host, port);
            if (await Task.WhenAny(connectTask, Task.Delay(timeoutMs)) != connectTask)
                return false;

            var stream = client.GetStream();
            var request = Encoding.UTF8.GetBytes("{\"cmd\":\"PING\"}\n");
            await stream.WriteAsync(request);

            var buffer = new byte[4096];
            stream.ReadTimeout = timeoutMs;
            var bytesRead = await stream.ReadAsync(buffer);
            var response = Encoding.UTF8.GetString(buffer, 0, bytesRead);
            return response.Contains("\"ok\":true") || response.Contains("\"ok\": true");
        }
        catch (Exception)
        {
            return false;
        }
    }

    public void Dispose()
    {
        _httpClient.Dispose();
    }
}
