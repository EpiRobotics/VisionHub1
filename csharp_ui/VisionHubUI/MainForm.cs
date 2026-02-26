using System.ComponentModel;
using VisionHubUI.Models;
using VisionHubUI.Services;

namespace VisionHubUI;

/// <summary>
/// Main form for VisionHub management UI.
/// Layout:
/// - Left panel: Project list with search + service controls
/// - Right panel: TabControl with per-project detail pages
/// - Bottom status bar: Service health, connection status
/// </summary>
public partial class MainForm : Form
{
    private readonly VisionHubApiClient _apiClient;
    private System.Windows.Forms.Timer _healthTimer;
    private System.Windows.Forms.Timer _refreshTimer;
    private List<ProjectInfo> _projects = new();
    private string? _selectedProjectId;

    // --- Left panel controls ---
    private Panel _leftPanel = null!;
    private TextBox _searchBox = null!;
    private ListBox _projectListBox = null!;
    private Button _btnRefresh = null!;
    private Button _btnReloadAll = null!;
    private Label _lblServiceUrl = null!;
    private TextBox _txtServiceUrl = null!;
    private Button _btnConnect = null!;

    // --- Right panel controls ---
    private TabControl _tabControl = null!;

    // --- Status bar ---
    private StatusStrip _statusStrip = null!;
    private ToolStripStatusLabel _lblHealthStatus = null!;
    private ToolStripStatusLabel _lblProjectCount = null!;
    private ToolStripStatusLabel _lblUptime = null!;

    public MainForm()
    {
        _apiClient = new VisionHubApiClient("http://localhost:8100");
        InitializeComponents();
        SetupTimers();
    }

    private void InitializeComponents()
    {
        // Form settings
        Text = "VisionHub - Industrial Visual Inspection Platform";
        Size = new Size(1280, 800);
        MinimumSize = new Size(960, 600);
        StartPosition = FormStartPosition.CenterScreen;

        // ============================================================
        // Left Panel - Project List & Service Controls
        // ============================================================
        _leftPanel = new Panel
        {
            Dock = DockStyle.Left,
            Width = 300,
            Padding = new Padding(8),
            BackColor = Color.FromArgb(245, 245, 250)
        };

        // Service URL
        _lblServiceUrl = new Label
        {
            Text = "Service URL:",
            Location = new Point(8, 8),
            Size = new Size(280, 20),
            Font = new Font("Segoe UI", 9, FontStyle.Bold)
        };
        _leftPanel.Controls.Add(_lblServiceUrl);

        _txtServiceUrl = new TextBox
        {
            Text = "http://localhost:8100",
            Location = new Point(8, 30),
            Size = new Size(200, 25)
        };
        _leftPanel.Controls.Add(_txtServiceUrl);

        _btnConnect = new Button
        {
            Text = "Connect",
            Location = new Point(214, 29),
            Size = new Size(70, 26),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(0, 122, 204),
            ForeColor = Color.White
        };
        _btnConnect.Click += BtnConnect_Click;
        _leftPanel.Controls.Add(_btnConnect);

        // Separator
        var sep1 = new Label
        {
            BorderStyle = BorderStyle.Fixed3D,
            Location = new Point(8, 62),
            Size = new Size(276, 2)
        };
        _leftPanel.Controls.Add(sep1);

        // Search box
        _searchBox = new TextBox
        {
            PlaceholderText = "Search projects...",
            Location = new Point(8, 72),
            Size = new Size(276, 25)
        };
        _searchBox.TextChanged += SearchBox_TextChanged;
        _leftPanel.Controls.Add(_searchBox);

        // Project list
        _projectListBox = new ListBox
        {
            Location = new Point(8, 102),
            Size = new Size(276, 500),
            Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
            Font = new Font("Segoe UI", 9.5f),
            IntegralHeight = false
        };
        _projectListBox.SelectedIndexChanged += ProjectListBox_SelectedIndexChanged;
        _leftPanel.Controls.Add(_projectListBox);

        // Buttons
        _btnRefresh = new Button
        {
            Text = "Refresh",
            Size = new Size(134, 32),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(0, 122, 204),
            ForeColor = Color.White,
            Anchor = AnchorStyles.Bottom | AnchorStyles.Left
        };
        _btnRefresh.Location = new Point(8, _leftPanel.Height - _btnRefresh.Height - 8);
        _btnRefresh.Click += BtnRefresh_Click;
        _leftPanel.Controls.Add(_btnRefresh);

        _btnReloadAll = new Button
        {
            Text = "Reload All",
            Size = new Size(134, 32),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(108, 117, 125),
            ForeColor = Color.White,
            Anchor = AnchorStyles.Bottom | AnchorStyles.Left
        };
        _btnReloadAll.Location = new Point(150, _leftPanel.Height - _btnReloadAll.Height - 8);
        _btnReloadAll.Click += BtnReloadAll_Click;
        _leftPanel.Controls.Add(_btnReloadAll);

        Controls.Add(_leftPanel);

        // ============================================================
        // Right Panel - Project Detail Tabs
        // ============================================================
        _tabControl = new TabControl
        {
            Dock = DockStyle.Fill,
            Font = new Font("Segoe UI", 9.5f)
        };
        Controls.Add(_tabControl);

        // ============================================================
        // Status Bar
        // ============================================================
        _statusStrip = new StatusStrip();
        _lblHealthStatus = new ToolStripStatusLabel("Service: Disconnected")
        {
            Spring = false,
            ForeColor = Color.Red
        };
        _lblProjectCount = new ToolStripStatusLabel("Projects: 0");
        _lblUptime = new ToolStripStatusLabel("")
        {
            Spring = true,
            TextAlign = System.Drawing.ContentAlignment.MiddleRight
        };
        _statusStrip.Items.AddRange(new ToolStripItem[]
        {
            _lblHealthStatus,
            new ToolStripSeparator(),
            _lblProjectCount,
            new ToolStripSeparator(),
            _lblUptime
        });
        Controls.Add(_statusStrip);

        // Ensure correct z-order (left panel on top of tab control)
        _leftPanel.BringToFront();
    }

    private void SetupTimers()
    {
        // Health check every 5 seconds
        _healthTimer = new System.Windows.Forms.Timer { Interval = 5000 };
        _healthTimer.Tick += async (s, e) => await CheckHealthAsync();
        _healthTimer.Start();

        // Refresh project list every 10 seconds
        _refreshTimer = new System.Windows.Forms.Timer { Interval = 10000 };
        _refreshTimer.Tick += async (s, e) => await RefreshProjectsAsync();
    }

    protected override async void OnShown(EventArgs e)
    {
        base.OnShown(e);
        await CheckHealthAsync();
        await RefreshProjectsAsync();
        _refreshTimer.Start();
    }

    // ==================================================================
    // Event Handlers
    // ==================================================================

    private async void BtnConnect_Click(object? sender, EventArgs e)
    {
        _apiClient.BaseUrl = _txtServiceUrl.Text.Trim();
        _btnConnect.Enabled = false;
        _btnConnect.Text = "...";
        await CheckHealthAsync();
        await RefreshProjectsAsync();
        _btnConnect.Enabled = true;
        _btnConnect.Text = "Connect";
    }

    private async void BtnRefresh_Click(object? sender, EventArgs e)
    {
        _btnRefresh.Enabled = false;
        await RefreshProjectsAsync();
        _btnRefresh.Enabled = true;
    }

    private async void BtnReloadAll_Click(object? sender, EventArgs e)
    {
        _btnReloadAll.Enabled = false;
        var success = await _apiClient.ReloadAllProjectsAsync();
        if (success)
        {
            await RefreshProjectsAsync();
        }
        else
        {
            MessageBox.Show("Failed to reload projects. Is the service running?",
                "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
        _btnReloadAll.Enabled = true;
    }

    private void SearchBox_TextChanged(object? sender, EventArgs e)
    {
        UpdateProjectListDisplay();
    }

    private void ProjectListBox_SelectedIndexChanged(object? sender, EventArgs e)
    {
        if (_projectListBox.SelectedIndex < 0) return;

        // Find matching project from filtered list
        var displayText = _projectListBox.SelectedItem?.ToString() ?? "";
        var project = _projects.FirstOrDefault(p => p.ToString() == displayText);
        if (project != null)
        {
            _selectedProjectId = project.ProjectId;
            ShowProjectTab(project);
        }
    }

    // ==================================================================
    // Service Health
    // ==================================================================

    private async Task CheckHealthAsync()
    {
        var health = await _apiClient.GetHealthAsync();
        if (health != null && health.Ok)
        {
            _lblHealthStatus.Text = $"Service: Online (v{health.Version})";
            _lblHealthStatus.ForeColor = Color.Green;
            _lblProjectCount.Text = $"Projects: {health.ProjectsCount}";
            var uptime = TimeSpan.FromSeconds(health.UptimeSeconds);
            _lblUptime.Text = $"Uptime: {uptime:hh\\:mm\\:ss} | GPU Workers: {health.GpuWorkers} | {health.Timestamp}";
        }
        else
        {
            _lblHealthStatus.Text = "Service: Disconnected";
            _lblHealthStatus.ForeColor = Color.Red;
            _lblProjectCount.Text = "Projects: -";
            _lblUptime.Text = "";
        }
    }

    // ==================================================================
    // Project List
    // ==================================================================

    private async Task RefreshProjectsAsync()
    {
        _projects = await _apiClient.GetProjectsAsync();
        UpdateProjectListDisplay();
    }

    private void UpdateProjectListDisplay()
    {
        var searchText = _searchBox.Text.Trim().ToLower();
        var filtered = string.IsNullOrEmpty(searchText)
            ? _projects
            : _projects.Where(p =>
                p.ProjectId.ToLower().Contains(searchText) ||
                p.DisplayName.ToLower().Contains(searchText)).ToList();

        var selectedText = _projectListBox.SelectedItem?.ToString();
        _projectListBox.BeginUpdate();
        _projectListBox.Items.Clear();
        foreach (var project in filtered)
        {
            _projectListBox.Items.Add(project.ToString());
        }
        _projectListBox.EndUpdate();

        // Restore selection
        if (selectedText != null)
        {
            var idx = _projectListBox.Items.IndexOf(selectedText);
            if (idx >= 0) _projectListBox.SelectedIndex = idx;
        }
    }

    // ==================================================================
    // Project Tab
    // ==================================================================

    private void ShowProjectTab(ProjectInfo project)
    {
        // Check if tab already exists
        var tabName = $"tab_{project.ProjectId}";
        TabPage? existing = null;
        foreach (TabPage tab in _tabControl.TabPages)
        {
            if (tab.Name == tabName)
            {
                existing = tab;
                break;
            }
        }

        if (existing != null)
        {
            _tabControl.SelectedTab = existing;
            UpdateProjectTabContent(existing, project);
            return;
        }

        // Create new tab
        var tabPage = new TabPage(project.ProjectId)
        {
            Name = tabName,
            AutoScroll = true,
            Padding = new Padding(12)
        };

        BuildProjectTabContent(tabPage, project);
        _tabControl.TabPages.Add(tabPage);
        _tabControl.SelectedTab = tabPage;
    }

    private void BuildProjectTabContent(TabPage tab, ProjectInfo project)
    {
        tab.Controls.Clear();
        int y = 12;

        // --- Section: Basic Info ---
        var lblTitle = new Label
        {
            Text = $"{project.DisplayName}",
            Font = new Font("Segoe UI", 14, FontStyle.Bold),
            Location = new Point(12, y),
            AutoSize = true
        };
        tab.Controls.Add(lblTitle);
        y += 35;

        var lblInfo = new Label
        {
            Text = $"Project ID: {project.ProjectId}    |    Port: {project.TcpPort}    |    Algorithm: {project.Algo}",
            Font = new Font("Segoe UI", 10),
            Location = new Point(12, y),
            AutoSize = true,
            ForeColor = Color.DimGray
        };
        tab.Controls.Add(lblInfo);
        y += 28;

        // Enabled toggle
        var chkEnabled = new CheckBox
        {
            Text = "Enabled",
            Checked = project.Enabled,
            Location = new Point(12, y),
            AutoSize = true,
            Font = new Font("Segoe UI", 10),
            Tag = project.ProjectId
        };
        chkEnabled.CheckedChanged += async (s, e) =>
        {
            if (s is CheckBox cb && cb.Tag is string pid)
            {
                await _apiClient.EnableProjectAsync(pid, cb.Checked);
            }
        };
        tab.Controls.Add(chkEnabled);
        y += 32;

        // Model status
        var modelStatus = project.ModelLoaded
            ? $"Model: {project.ModelVersion} (Loaded)"
            : "Model: Not Loaded";
        var modelColor = project.ModelLoaded ? Color.Green : Color.OrangeRed;
        var lblModel = new Label
        {
            Text = modelStatus,
            Font = new Font("Segoe UI", 10, FontStyle.Bold),
            ForeColor = modelColor,
            Location = new Point(12, y),
            AutoSize = true,
            Name = "lblModel"
        };
        tab.Controls.Add(lblModel);

        // Load model button
        var btnLoadModel = new Button
        {
            Text = "Load Model",
            Size = new Size(100, 28),
            Location = new Point(350, y - 3),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(40, 167, 69),
            ForeColor = Color.White,
            Tag = project.ProjectId
        };
        btnLoadModel.Click += async (s, e) =>
        {
            if (s is Button btn && btn.Tag is string pid)
            {
                btn.Enabled = false;
                btn.Text = "Loading...";
                var success = await _apiClient.LoadModelAsync(pid);
                btn.Text = success ? "Loaded!" : "Failed";
                btn.Enabled = true;
                await RefreshProjectsAsync();
            }
        };
        tab.Controls.Add(btnLoadModel);
        y += 35;

        // TCP status
        var tcpStatus = project.TcpRunning ? "TCP: Running" : "TCP: Stopped";
        var tcpColor = project.TcpRunning ? Color.Green : Color.Red;
        var lblTcp = new Label
        {
            Text = tcpStatus,
            Font = new Font("Segoe UI", 10),
            ForeColor = tcpColor,
            Location = new Point(12, y),
            AutoSize = true
        };
        tab.Controls.Add(lblTcp);

        // TCP ping button
        var btnPingTcp = new Button
        {
            Text = "Ping TCP",
            Size = new Size(80, 28),
            Location = new Point(350, y - 3),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(108, 117, 125),
            ForeColor = Color.White,
            Tag = project.TcpPort
        };
        btnPingTcp.Click += async (s, e) =>
        {
            if (s is Button btn && btn.Tag is int port)
            {
                btn.Text = "...";
                var ok = await _apiClient.PingTcpPortAsync("127.0.0.1", port);
                btn.Text = ok ? "OK!" : "Fail";
                await Task.Delay(1500);
                btn.Text = "Ping TCP";
            }
        };
        tab.Controls.Add(btnPingTcp);
        y += 40;

        // Separator
        var sep = new Label
        {
            BorderStyle = BorderStyle.Fixed3D,
            Location = new Point(12, y),
            Size = new Size(700, 2)
        };
        tab.Controls.Add(sep);
        y += 12;

        // --- Section: Statistics ---
        var lblStatsHeader = new Label
        {
            Text = "Statistics",
            Font = new Font("Segoe UI", 12, FontStyle.Bold),
            Location = new Point(12, y),
            AutoSize = true
        };
        tab.Controls.Add(lblStatsHeader);
        y += 28;

        var stats = project.Stats;
        var statsText = stats != null
            ? $"Total: {stats.TotalJobs}    OK: {stats.OkCount}    NG: {stats.NgCount}    " +
              $"Error: {stats.ErrorCount}    Avg Infer: {stats.AvgInferMs:F1}ms    Last: {stats.LastResultTime}"
            : "No statistics available";

        var lblStats = new Label
        {
            Text = statsText,
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(12, y),
            AutoSize = true,
            Name = "lblStats"
        };
        tab.Controls.Add(lblStats);
        y += 30;

        // Separator
        var sep2 = new Label
        {
            BorderStyle = BorderStyle.Fixed3D,
            Location = new Point(12, y),
            Size = new Size(700, 2)
        };
        tab.Controls.Add(sep2);
        y += 12;

        // --- Section: Training ---
        var lblTrainHeader = new Label
        {
            Text = "Training",
            Font = new Font("Segoe UI", 12, FontStyle.Bold),
            Location = new Point(12, y),
            AutoSize = true
        };
        tab.Controls.Add(lblTrainHeader);
        y += 28;

        var btnStartTrain = new Button
        {
            Text = "Start Training",
            Size = new Size(120, 32),
            Location = new Point(12, y),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(0, 123, 255),
            ForeColor = Color.White,
            Tag = project.ProjectId
        };

        var lblTrainStatus = new Label
        {
            Text = "",
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(145, y + 6),
            AutoSize = true,
            Name = "lblTrainStatus"
        };

        var txtTrainLog = new TextBox
        {
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
            Location = new Point(12, y + 40),
            Size = new Size(700, 120),
            Font = new Font("Consolas", 9),
            Name = "txtTrainLog"
        };

        btnStartTrain.Click += async (s, e) =>
        {
            if (s is Button btn && btn.Tag is string pid)
            {
                btn.Enabled = false;
                btn.Text = "Training...";
                lblTrainStatus.Text = "Starting...";

                var resp = await _apiClient.StartTrainingAsync(pid);
                if (resp != null && resp.Ok)
                {
                    // Poll training status
                    var trainJobId = resp.TrainJobId;
                    while (true)
                    {
                        await Task.Delay(2000);
                        var status = await _apiClient.GetTrainStatusAsync(trainJobId);
                        if (status == null) break;

                        lblTrainStatus.Text = $"[{status.Progress:F0}%] {status.Status} - {status.Message}";
                        txtTrainLog.Text = string.Join(Environment.NewLine, status.LogLines);
                        txtTrainLog.SelectionStart = txtTrainLog.Text.Length;
                        txtTrainLog.ScrollToCaret();

                        if (status.Status == "completed" || status.Status == "failed")
                        {
                            break;
                        }
                    }
                }
                else
                {
                    lblTrainStatus.Text = "Failed to start training";
                }

                btn.Enabled = true;
                btn.Text = "Start Training";
            }
        };

        tab.Controls.Add(btnStartTrain);
        tab.Controls.Add(lblTrainStatus);
        tab.Controls.Add(txtTrainLog);
        y += 170;

        // Separator
        var sep3 = new Label
        {
            BorderStyle = BorderStyle.Fixed3D,
            Location = new Point(12, y),
            Size = new Size(700, 2)
        };
        tab.Controls.Add(sep3);
        y += 12;

        // --- Section: Test Inference ---
        var lblTestHeader = new Label
        {
            Text = "Test Inference",
            Font = new Font("Segoe UI", 12, FontStyle.Bold),
            Location = new Point(12, y),
            AutoSize = true
        };
        tab.Controls.Add(lblTestHeader);
        y += 28;

        var txtImagePath = new TextBox
        {
            PlaceholderText = "Enter image path or browse...",
            Location = new Point(12, y),
            Size = new Size(500, 25),
            Name = "txtImagePath"
        };
        tab.Controls.Add(txtImagePath);

        var btnBrowse = new Button
        {
            Text = "Browse",
            Size = new Size(70, 26),
            Location = new Point(518, y - 1),
            FlatStyle = FlatStyle.Flat
        };
        btnBrowse.Click += (s, e) =>
        {
            using var ofd = new OpenFileDialog
            {
                Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff|All Files|*.*"
            };
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                txtImagePath.Text = ofd.FileName;
            }
        };
        tab.Controls.Add(btnBrowse);

        var btnRunInfer = new Button
        {
            Text = "Run",
            Size = new Size(60, 26),
            Location = new Point(594, y - 1),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(40, 167, 69),
            ForeColor = Color.White,
            Tag = project.ProjectId
        };

        var txtInferResult = new TextBox
        {
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
            Location = new Point(12, y + 32),
            Size = new Size(700, 100),
            Font = new Font("Consolas", 9),
            Name = "txtInferResult"
        };

        btnRunInfer.Click += async (s, e) =>
        {
            if (s is Button btn && btn.Tag is string pid)
            {
                var imagePath = txtImagePath.Text.Trim();
                if (string.IsNullOrEmpty(imagePath))
                {
                    MessageBox.Show("Please enter an image path.", "Warning",
                        MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                btn.Enabled = false;
                btn.Text = "...";
                txtInferResult.Text = "Running inference...";

                var result = await _apiClient.RunInferenceAsync(pid, imagePath);
                if (result != null)
                {
                    var timing = result.TimingMs;
                    txtInferResult.Text =
                        $"Pred: {result.Pred}    Score: {result.Score:F4}    Threshold: {result.Threshold:F4}\r\n" +
                        $"Timing: total={timing?.Total:F1}ms  infer={timing?.Infer:F1}ms  " +
                        $"post={timing?.Post:F1}ms  save={timing?.Save:F1}ms\r\n" +
                        $"Model: {result.ModelVersion}\r\n" +
                        $"Regions: {result.Regions?.Count ?? 0}";

                    if (result.Error != null)
                    {
                        txtInferResult.Text += $"\r\nERROR: [{result.Error.Code}] {result.Error.Message}";
                    }
                }
                else
                {
                    txtInferResult.Text = "Failed to run inference. Check service connection.";
                }

                btn.Enabled = true;
                btn.Text = "Run";
            }
        };

        tab.Controls.Add(btnRunInfer);
        tab.Controls.Add(txtInferResult);
    }

    private void UpdateProjectTabContent(TabPage tab, ProjectInfo project)
    {
        // Update dynamic labels
        var lblModel = tab.Controls.Find("lblModel", true).FirstOrDefault() as Label;
        if (lblModel != null)
        {
            lblModel.Text = project.ModelLoaded
                ? $"Model: {project.ModelVersion} (Loaded)"
                : "Model: Not Loaded";
            lblModel.ForeColor = project.ModelLoaded ? Color.Green : Color.OrangeRed;
        }

        var lblStats = tab.Controls.Find("lblStats", true).FirstOrDefault() as Label;
        if (lblStats != null && project.Stats != null)
        {
            var stats = project.Stats;
            lblStats.Text = $"Total: {stats.TotalJobs}    OK: {stats.OkCount}    NG: {stats.NgCount}    " +
                            $"Error: {stats.ErrorCount}    Avg Infer: {stats.AvgInferMs:F1}ms    Last: {stats.LastResultTime}";
        }
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        _healthTimer.Stop();
        _refreshTimer.Stop();
        _apiClient.Dispose();
        base.OnFormClosing(e);
    }
}
