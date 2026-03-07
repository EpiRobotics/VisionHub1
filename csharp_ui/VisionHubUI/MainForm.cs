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

    // Per-project log state: project_id -> next_index for incremental fetch
    private readonly Dictionary<string, int> _logNextIndex = new();
    // Per-project log auto-refresh timers
    private readonly Dictionary<string, System.Windows.Forms.Timer> _logTimers = new();

    // Suppress tab switching during automatic timer-driven refreshes
    private bool _suppressTabSwitch = false;

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

        // Ensure correct z-order: Fill control must be highest z-order
        // so it fills remaining space AFTER Left-docked panel is placed.
        _tabControl.BringToFront();

        // ============================================================
        // Permanent Tab: Label Training Workflow
        // ============================================================
        BuildLabelTrainingTab();
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
            // Only switch tab on manual clicks, not during timer-driven refresh
            if (!_suppressTabSwitch)
            {
                ShowProjectTab(project);
            }
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
        _suppressTabSwitch = true;
        try
        {
            _projects = await _apiClient.GetProjectsAsync();
            UpdateProjectListDisplay();
        }
        finally
        {
            _suppressTabSwitch = false;
        }
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
            Padding = new Padding(0)
        };

        BuildProjectTabContent(tabPage, project);
        _tabControl.TabPages.Add(tabPage);
        _tabControl.SelectedTab = tabPage;
    }

    private void BuildProjectTabContent(TabPage tab, ProjectInfo project)
    {
        tab.Controls.Clear();

        // Use a scrollable panel for all content
        var contentPanel = new Panel
        {
            Dock = DockStyle.Fill,
            AutoScroll = true,
            Padding = new Padding(16, 12, 16, 12)
        };
        tab.Controls.Add(contentPanel);

        // Calculate usable width (will be updated on resize)
        int contentWidth = Math.Max(contentPanel.ClientSize.Width - 40, 600);
        int leftMargin = 16;
        int y = 8;

        // ===== Section: Basic Info =====
        var lblTitle = new Label
        {
            Text = project.DisplayName,
            Font = new Font("Segoe UI", 14, FontStyle.Bold),
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 30)
        };
        contentPanel.Controls.Add(lblTitle);
        y += 36;

        var lblInfo = new Label
        {
            Text = $"Project ID: {project.ProjectId}   |   Port: {project.TcpPort}   |   Algorithm: {project.Algo}",
            Font = new Font("Segoe UI", 10),
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 22),
            ForeColor = Color.DimGray
        };
        contentPanel.Controls.Add(lblInfo);
        y += 28;

        // Enabled toggle + Model status row
        var chkEnabled = new CheckBox
        {
            Text = "Enabled",
            Checked = project.Enabled,
            Location = new Point(leftMargin, y),
            Size = new Size(100, 24),
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
        contentPanel.Controls.Add(chkEnabled);

        var modelStatus = project.ModelLoaded
            ? $"Model: {project.ModelVersion} (Loaded)"
            : "Model: Not Loaded";
        var modelColor = project.ModelLoaded ? Color.Green : Color.OrangeRed;
        var lblModel = new Label
        {
            Text = modelStatus,
            Font = new Font("Segoe UI", 10, FontStyle.Bold),
            ForeColor = modelColor,
            Location = new Point(leftMargin + 110, y + 2),
            Size = new Size(300, 22),
            Name = "lblModel"
        };
        contentPanel.Controls.Add(lblModel);
        y += 32;

        // Action buttons row
        var flowButtons = new FlowLayoutPanel
        {
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 36),
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false,
            AutoSize = false
        };

        var btnLoadModel = new Button
        {
            Text = "Load Model",
            Size = new Size(110, 30),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(40, 167, 69),
            ForeColor = Color.White,
            Margin = new Padding(0, 0, 8, 0),
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
        flowButtons.Controls.Add(btnLoadModel);

        var btnPingTcp = new Button
        {
            Text = "Ping TCP",
            Size = new Size(90, 30),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(108, 117, 125),
            ForeColor = Color.White,
            Margin = new Padding(0, 0, 8, 0),
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
        flowButtons.Controls.Add(btnPingTcp);

        var tcpStatus = project.TcpRunning ? "TCP: Running" : "TCP: Stopped";
        var tcpColor = project.TcpRunning ? Color.Green : Color.Red;
        var lblTcp = new Label
        {
            Text = tcpStatus,
            Font = new Font("Segoe UI", 10),
            ForeColor = tcpColor,
            Size = new Size(120, 26),
            TextAlign = ContentAlignment.MiddleLeft,
            Margin = new Padding(8, 4, 0, 0)
        };
        flowButtons.Controls.Add(lblTcp);

        contentPanel.Controls.Add(flowButtons);
        y += 42;

        // ===== Separator =====
        AddSeparator(contentPanel, leftMargin, ref y, contentWidth);

        // ===== Section: Statistics =====
        AddSectionHeader(contentPanel, "Statistics", leftMargin, ref y);

        var stats = project.Stats;
        var statsText = stats != null
            ? $"Total: {stats.TotalJobs}   OK: {stats.OkCount}   NG: {stats.NgCount}   " +
              $"Error: {stats.ErrorCount}   Avg Infer: {stats.AvgInferMs:F1}ms   Last: {stats.LastResultTime}"
            : "No statistics available";

        var lblStats = new Label
        {
            Text = statsText,
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 22),
            Name = "lblStats"
        };
        contentPanel.Controls.Add(lblStats);
        y += 28;

        // ===== Separator =====
        AddSeparator(contentPanel, leftMargin, ref y, contentWidth);

        // ===== Section: Training =====
        AddSectionHeader(contentPanel, "Training", leftMargin, ref y);

        var btnStartTrain = new Button
        {
            Text = "Start Training",
            Size = new Size(130, 32),
            Location = new Point(leftMargin, y),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(0, 123, 255),
            ForeColor = Color.White,
            Tag = project.ProjectId
        };

        var lblTrainStatus = new Label
        {
            Text = "",
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(leftMargin + 140, y + 6),
            Size = new Size(contentWidth - 150, 22),
            Name = "lblTrainStatus"
        };

        var txtTrainLog = new TextBox
        {
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
            Location = new Point(leftMargin, y + 40),
            Size = new Size(contentWidth, 110),
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

        contentPanel.Controls.Add(btnStartTrain);
        contentPanel.Controls.Add(lblTrainStatus);
        contentPanel.Controls.Add(txtTrainLog);
        y += 160;

        // ===== Separator =====
        AddSeparator(contentPanel, leftMargin, ref y, contentWidth);

        // ===== Section: Test Inference =====
        AddSectionHeader(contentPanel, "Test Inference", leftMargin, ref y);

        var txtImagePath = new TextBox
        {
            PlaceholderText = "Enter image path or browse...",
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth - 160, 25),
            Name = "txtImagePath"
        };
        contentPanel.Controls.Add(txtImagePath);

        var btnBrowse = new Button
        {
            Text = "Browse",
            Size = new Size(70, 26),
            Location = new Point(leftMargin + contentWidth - 150, y - 1),
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
        contentPanel.Controls.Add(btnBrowse);

        var btnRunInfer = new Button
        {
            Text = "Run",
            Size = new Size(70, 26),
            Location = new Point(leftMargin + contentWidth - 72, y - 1),
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
            Location = new Point(leftMargin, y + 32),
            Size = new Size(contentWidth, 100),
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

        contentPanel.Controls.Add(btnRunInfer);
        contentPanel.Controls.Add(txtInferResult);
        y += 140;

        // ===== Separator =====
        AddSeparator(contentPanel, leftMargin, ref y, contentWidth);

        // ===== Section: Overlay Output Path =====
        AddSectionHeader(contentPanel, "Overlay Output Path", leftMargin, ref y);

        var lblOverlayDesc = new Label
        {
            Text = "Set a fixed overlay output directory. Overlay is saved as output.jpg (overwrites each inference).\n" +
                   "External vision software can monitor this file for OK/NG results. Leave empty for per-job artifacts.",
            Font = new Font("Segoe UI", 9f),
            ForeColor = Color.DimGray,
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 32)
        };
        contentPanel.Controls.Add(lblOverlayDesc);
        y += 36;

        var txtOverlayDir = new TextBox
        {
            PlaceholderText = "e.g. D:\\results (overlay saved as output.jpg)",
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth - 240, 25),
            Name = "txtOverlayDir"
        };
        contentPanel.Controls.Add(txtOverlayDir);

        var btnBrowseOverlay = new Button
        {
            Text = "Browse...",
            Size = new Size(80, 26),
            Location = new Point(leftMargin + contentWidth - 230, y - 1),
            FlatStyle = FlatStyle.Flat
        };
        btnBrowseOverlay.Click += (s, e) =>
        {
            using var fbd = new FolderBrowserDialog { Description = "Select overlay output directory" };
            if (fbd.ShowDialog() == DialogResult.OK)
                txtOverlayDir.Text = fbd.SelectedPath;
        };
        contentPanel.Controls.Add(btnBrowseOverlay);

        var btnApplyOverlay = new Button
        {
            Text = "Apply",
            Size = new Size(70, 26),
            Location = new Point(leftMargin + contentWidth - 140, y - 1),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(0, 123, 255),
            ForeColor = Color.White
        };
        contentPanel.Controls.Add(btnApplyOverlay);

        var lblOverlayResult = new Label
        {
            Text = "",
            Font = new Font("Segoe UI", 9f),
            Location = new Point(leftMargin, y + 28),
            Size = new Size(contentWidth, 20),
            Name = "lblOverlayResult"
        };
        contentPanel.Controls.Add(lblOverlayResult);

        btnApplyOverlay.Click += async (s, e) =>
        {
            var overlayDir = txtOverlayDir.Text.Trim();
            var overlayFullPath = string.IsNullOrEmpty(overlayDir)
                ? ""
                : Path.Combine(overlayDir, "output.jpg");

            btnApplyOverlay.Enabled = false;
            var ok = await _apiClient.SetOverlayPathAsync(project.ProjectId, overlayFullPath);
            if (ok)
            {
                lblOverlayResult.Text = string.IsNullOrEmpty(overlayFullPath)
                    ? "Cleared. Using per-job artifacts mode. (saved to config)"
                    : $"Set: {overlayFullPath} (saved to config, persists after restart)";
                lblOverlayResult.ForeColor = Color.Green;
            }
            else
            {
                lblOverlayResult.Text = "Failed. Check service connection.";
                lblOverlayResult.ForeColor = Color.Red;
            }
            btnApplyOverlay.Enabled = true;
        };
        y += 54;

        // ===== Separator =====
        AddSeparator(contentPanel, leftMargin, ref y, contentWidth);

        // ===== Section: NG Threshold =====
        AddSectionHeader(contentPanel, "NG Threshold", leftMargin, ref y);

        var lblThrDesc2 = new Label
        {
            Text = "Set global NG threshold. Score >= threshold is NG. Leave empty or 0 for per-class trained thresholds.",
            Font = new Font("Segoe UI", 9f),
            ForeColor = Color.DimGray,
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 18)
        };
        contentPanel.Controls.Add(lblThrDesc2);
        y += 22;

        var lblThrInput2 = new Label
        {
            Text = "Threshold:",
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(leftMargin, y + 2),
            Size = new Size(80, 20)
        };
        contentPanel.Controls.Add(lblThrInput2);

        var txtThrGlobal2 = new TextBox
        {
            PlaceholderText = "e.g. 2.50",
            Location = new Point(leftMargin + 85, y),
            Size = new Size(120, 25),
            Name = "txtThrGlobal"
        };
        contentPanel.Controls.Add(txtThrGlobal2);

        var btnApplyThr2 = new Button
        {
            Text = "Apply",
            Size = new Size(70, 26),
            Location = new Point(leftMargin + 215, y - 1),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(0, 123, 255),
            ForeColor = Color.White
        };
        contentPanel.Controls.Add(btnApplyThr2);

        var lblThrResult2 = new Label
        {
            Text = "",
            Font = new Font("Segoe UI", 9f),
            Location = new Point(leftMargin + 295, y + 2),
            Size = new Size(contentWidth - 295, 20),
            Name = "lblThrResult"
        };
        contentPanel.Controls.Add(lblThrResult2);

        btnApplyThr2.Click += async (s, e) =>
        {
            double? thrValue = null;
            var thrText = txtThrGlobal2.Text.Trim();
            if (!string.IsNullOrEmpty(thrText))
            {
                if (double.TryParse(thrText, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out double parsed) && parsed > 0)
                {
                    thrValue = parsed;
                }
                else
                {
                    MessageBox.Show("Invalid threshold. Use a positive number (e.g. 2.50).",
                        "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
            }

            btnApplyThr2.Enabled = false;
            var ok = await _apiClient.SetThresholdAsync(project.ProjectId, thrValue);
            if (ok)
            {
                lblThrResult2.Text = thrValue.HasValue
                    ? $"Set to {thrValue.Value:F2} (saved to config)"
                    : "Using per-class thresholds (saved to config)";
                lblThrResult2.ForeColor = Color.Green;
            }
            else
            {
                lblThrResult2.Text = "Failed. Check service connection.";
                lblThrResult2.ForeColor = Color.Red;
            }
            btnApplyThr2.Enabled = true;
        };
        y += 32;

        // ===== Separator =====
        AddSeparator(contentPanel, leftMargin, ref y, contentWidth);

        // ===== Section: Log =====
        AddSectionHeader(contentPanel, "Log", leftMargin, ref y);

        var flowLogButtons = new FlowLayoutPanel
        {
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 32),
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false,
            AutoSize = false
        };

        var chkAutoRefresh = new CheckBox
        {
            Text = "Auto Refresh (2s)",
            Checked = true,
            Size = new Size(140, 24),
            Font = new Font("Segoe UI", 9),
            Margin = new Padding(0, 3, 12, 0)
        };
        flowLogButtons.Controls.Add(chkAutoRefresh);

        var btnRefreshLog = new Button
        {
            Text = "Refresh",
            Size = new Size(80, 26),
            FlatStyle = FlatStyle.Flat,
            Margin = new Padding(0, 0, 8, 0)
        };
        flowLogButtons.Controls.Add(btnRefreshLog);

        var btnClearLog = new Button
        {
            Text = "Clear",
            Size = new Size(80, 26),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(220, 53, 69),
            ForeColor = Color.White,
            Margin = new Padding(0, 0, 8, 0)
        };
        flowLogButtons.Controls.Add(btnClearLog);

        contentPanel.Controls.Add(flowLogButtons);
        y += 36;

        var txtLog = new TextBox
        {
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Both,
            WordWrap = false,
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 180),
            Font = new Font("Consolas", 8.5f),
            BackColor = Color.FromArgb(30, 30, 30),
            ForeColor = Color.FromArgb(220, 220, 220),
            Name = "txtLog"
        };
        contentPanel.Controls.Add(txtLog);

        // Initialize log index tracking
        var pid = project.ProjectId;
        if (!_logNextIndex.ContainsKey(pid))
            _logNextIndex[pid] = 0;

        // Refresh log function
        async Task RefreshLogAsync()
        {
            var resp = await _apiClient.GetProjectLogsAsync(pid, _logNextIndex[pid]);
            if (resp != null && resp.Entries.Count > 0)
            {
                _logNextIndex[pid] = resp.NextIndex;
                var lines = resp.Entries.Select(e =>
                    $"[{e.Timestamp}] [{e.Level}] [{e.Source}] {e.Message}");
                txtLog.AppendText(string.Join(Environment.NewLine, lines) + Environment.NewLine);
            }
        }

        // Manual refresh
        btnRefreshLog.Click += async (s, e) => await RefreshLogAsync();

        // Clear log
        btnClearLog.Click += async (s, e) =>
        {
            await _apiClient.ClearProjectLogsAsync(pid);
            _logNextIndex[pid] = 0;
            txtLog.Clear();
        };

        // Auto-refresh timer
        var logTimer = new System.Windows.Forms.Timer { Interval = 2000 };
        logTimer.Tick += async (s, e) =>
        {
            if (chkAutoRefresh.Checked)
                await RefreshLogAsync();
        };
        logTimer.Start();

        // Track timer for cleanup
        if (_logTimers.ContainsKey(pid))
        {
            _logTimers[pid].Stop();
            _logTimers[pid].Dispose();
        }
        _logTimers[pid] = logTimer;

        chkAutoRefresh.CheckedChanged += (s, e) =>
        {
            if (chkAutoRefresh.Checked)
                logTimer.Start();
            else
                logTimer.Stop();
        };
    }

    // ==================================================================
    // Label Training Workflow Tab
    // ==================================================================

    private void BuildLabelTrainingTab()
    {
        var tab = new TabPage("Label Training")
        {
            Name = "tab_label_training",
            Padding = new Padding(0)
        };

        var contentPanel = new Panel
        {
            Dock = DockStyle.Fill,
            AutoScroll = true,
            Padding = new Padding(16, 12, 16, 12)
        };
        tab.Controls.Add(contentPanel);

        int contentWidth = Math.Max(700, 600);
        int leftMargin = 16;
        int y = 8;

        // ===== Title =====
        var lblTitle = new Label
        {
            Text = "Label (Glyph) Training Workflow",
            Font = new Font("Segoe UI", 14, FontStyle.Bold),
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 30)
        };
        contentPanel.Controls.Add(lblTitle);
        y += 32;

        var lblDesc = new Label
        {
            Text = "Step-by-step training pipeline for character/glyph anomaly detection using PatchCore.\n" +
                   "Step 1: Crop glyphs from images + JSON annotations into glyph_bank.\n" +
                   "Step 2: Review glyph_bank class statistics.\n" +
                   "Step 3: Train per-class PatchCore memory banks.",
            Font = new Font("Segoe UI", 9.5f),
            ForeColor = Color.DimGray,
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 60)
        };
        contentPanel.Controls.Add(lblDesc);
        y += 66;

        AddSeparator(contentPanel, leftMargin, ref y, contentWidth);

        // ===== Step 1: Crop Glyphs =====
        AddSectionHeader(contentPanel, "Step 1: Crop Glyphs from JSON", leftMargin, ref y);

        // Image Directory
        var lblImgDir = new Label
        {
            Text = "Image Directory (OK samples):",
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(leftMargin, y),
            Size = new Size(200, 20)
        };
        contentPanel.Controls.Add(lblImgDir);
        y += 22;

        var txtImgDir = new TextBox
        {
            PlaceholderText = @"e.g. E:\AIInspect\projects\label_check\datasets\ok",
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth - 90, 25),
            Name = "txtLabelImgDir"
        };
        contentPanel.Controls.Add(txtImgDir);

        var btnBrowseImgDir = new Button
        {
            Text = "Browse",
            Size = new Size(80, 26),
            Location = new Point(leftMargin + contentWidth - 82, y - 1),
            FlatStyle = FlatStyle.Flat
        };
        btnBrowseImgDir.Click += (s, e) =>
        {
            using var fbd = new FolderBrowserDialog { Description = "Select image directory (OK samples)" };
            if (fbd.ShowDialog() == DialogResult.OK)
                txtImgDir.Text = fbd.SelectedPath;
        };
        contentPanel.Controls.Add(btnBrowseImgDir);
        y += 32;

        // JSON Directory
        var lblJsonDir = new Label
        {
            Text = "JSON Annotation Directory:",
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(leftMargin, y),
            Size = new Size(200, 20)
        };
        contentPanel.Controls.Add(lblJsonDir);
        y += 22;

        var txtJsonDir = new TextBox
        {
            PlaceholderText = @"e.g. D:\glyph_bank\json",
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth - 90, 25),
            Name = "txtLabelJsonDir"
        };
        contentPanel.Controls.Add(txtJsonDir);

        var btnBrowseJsonDir = new Button
        {
            Text = "Browse",
            Size = new Size(80, 26),
            Location = new Point(leftMargin + contentWidth - 82, y - 1),
            FlatStyle = FlatStyle.Flat
        };
        btnBrowseJsonDir.Click += (s, e) =>
        {
            using var fbd = new FolderBrowserDialog { Description = "Select JSON annotation directory" };
            if (fbd.ShowDialog() == DialogResult.OK)
                txtJsonDir.Text = fbd.SelectedPath;
        };
        contentPanel.Controls.Add(btnBrowseJsonDir);
        y += 32;

        // Output (Glyph Bank) Directory
        var lblBankDir = new Label
        {
            Text = "Output Glyph Bank Directory:",
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(leftMargin, y),
            Size = new Size(200, 20)
        };
        contentPanel.Controls.Add(lblBankDir);
        y += 22;

        var txtBankDir = new TextBox
        {
            PlaceholderText = @"e.g. D:\glyph_bank\crops",
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth - 90, 25),
            Name = "txtLabelBankDir"
        };
        contentPanel.Controls.Add(txtBankDir);

        var btnBrowseBankDir = new Button
        {
            Text = "Browse",
            Size = new Size(80, 26),
            Location = new Point(leftMargin + contentWidth - 82, y - 1),
            FlatStyle = FlatStyle.Flat
        };
        btnBrowseBankDir.Click += (s, e) =>
        {
            using var fbd = new FolderBrowserDialog { Description = "Select glyph bank output directory" };
            if (fbd.ShowDialog() == DialogResult.OK)
                txtBankDir.Text = fbd.SelectedPath;
        };
        contentPanel.Controls.Add(btnBrowseBankDir);
        y += 32;

        // Pad setting
        var lblPad = new Label
        {
            Text = "Padding (px):",
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(leftMargin, y),
            Size = new Size(90, 20)
        };
        contentPanel.Controls.Add(lblPad);

        var nudPad = new NumericUpDown
        {
            Minimum = 0,
            Maximum = 20,
            Value = 2,
            Location = new Point(leftMargin + 95, y - 2),
            Size = new Size(60, 25)
        };
        contentPanel.Controls.Add(nudPad);

        // Crop button
        var btnCrop = new Button
        {
            Text = "Crop Glyphs",
            Size = new Size(130, 32),
            Location = new Point(leftMargin + 200, y - 4),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(0, 123, 255),
            ForeColor = Color.White
        };
        contentPanel.Controls.Add(btnCrop);
        y += 36;

        // Crop result label
        var lblCropResult = new Label
        {
            Text = "",
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 22),
            Name = "lblCropResult"
        };
        contentPanel.Controls.Add(lblCropResult);
        y += 26;

        // Class summary (DataGridView for glyph bank classes)
        var dgvClasses = new DataGridView
        {
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 150),
            ReadOnly = true,
            AllowUserToAddRows = false,
            AllowUserToDeleteRows = false,
            RowHeadersVisible = false,
            SelectionMode = DataGridViewSelectionMode.FullRowSelect,
            AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
            Font = new Font("Segoe UI", 9),
            Name = "dgvLabelClasses"
        };
        dgvClasses.Columns.Add("ClassName", "Character Class");
        dgvClasses.Columns.Add("Count", "Image Count");
        dgvClasses.Columns["Count"]!.DefaultCellStyle.Alignment = DataGridViewContentAlignment.MiddleRight;
        contentPanel.Controls.Add(dgvClasses);
        y += 158;

        // Crop button handler
        btnCrop.Click += async (s, e) =>
        {
            var imgDir = txtImgDir.Text.Trim();
            var jsonDir = txtJsonDir.Text.Trim();
            var bankDir = txtBankDir.Text.Trim();

            if (string.IsNullOrEmpty(imgDir) || string.IsNullOrEmpty(jsonDir) || string.IsNullOrEmpty(bankDir))
            {
                MessageBox.Show("Please fill in all three directories.", "Warning",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            btnCrop.Enabled = false;
            btnCrop.Text = "Cropping...";
            lblCropResult.Text = "Processing...";
            dgvClasses.Rows.Clear();

            var resp = await _apiClient.LabelCropGlyphsAsync(imgDir, jsonDir, bankDir, (int)nudPad.Value);
            if (resp != null && resp.Ok)
            {
                lblCropResult.Text = $"Cropped {resp.TotalCrops} glyphs from {resp.ProcessedFiles}/{resp.TotalJsonFiles} JSON files. " +
                                     $"Classes: {resp.Classes.Count}";
                lblCropResult.ForeColor = Color.Green;
                foreach (var cls in resp.Classes)
                    dgvClasses.Rows.Add(cls.ClassName, cls.Count);

                if (resp.Errors.Count > 0)
                {
                    lblCropResult.Text += $"  ({resp.Errors.Count} errors)";
                    lblCropResult.ForeColor = Color.OrangeRed;
                }
            }
            else
            {
                lblCropResult.Text = "Crop failed. Check service connection and paths.";
                lblCropResult.ForeColor = Color.Red;
            }

            btnCrop.Enabled = true;
            btnCrop.Text = "Crop Glyphs";
        };

        AddSeparator(contentPanel, leftMargin, ref y, contentWidth);

        // ===== Step 2: Scan/Review Bank =====
        AddSectionHeader(contentPanel, "Step 2: Review Glyph Bank", leftMargin, ref y);

        var flowScan = new FlowLayoutPanel
        {
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 34),
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false
        };

        var btnScanBank = new Button
        {
            Text = "Scan Bank",
            Size = new Size(110, 30),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(108, 117, 125),
            ForeColor = Color.White,
            Margin = new Padding(0, 0, 8, 0)
        };
        flowScan.Controls.Add(btnScanBank);

        var lblScanResult = new Label
        {
            Text = "(Uses the 'Output Glyph Bank Directory' from Step 1)",
            Font = new Font("Segoe UI", 9.5f),
            ForeColor = Color.DimGray,
            Size = new Size(contentWidth - 130, 26),
            TextAlign = ContentAlignment.MiddleLeft,
            Margin = new Padding(4, 4, 0, 0),
            Name = "lblScanResult"
        };
        flowScan.Controls.Add(lblScanResult);

        contentPanel.Controls.Add(flowScan);
        y += 38;

        btnScanBank.Click += async (s, e) =>
        {
            var bankDir = txtBankDir.Text.Trim();
            if (string.IsNullOrEmpty(bankDir))
            {
                MessageBox.Show("Please set the glyph bank directory in Step 1.", "Warning",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            btnScanBank.Enabled = false;
            btnScanBank.Text = "Scanning...";
            dgvClasses.Rows.Clear();

            var resp = await _apiClient.LabelScanBankAsync(bankDir);
            if (resp != null && resp.Ok)
            {
                lblScanResult.Text = $"Found {resp.TotalClasses} classes, {resp.TotalImages} total images";
                lblScanResult.ForeColor = Color.Green;
                foreach (var cls in resp.Classes)
                    dgvClasses.Rows.Add(cls.ClassName, cls.Count);
            }
            else
            {
                lblScanResult.Text = "Scan failed. Check path.";
                lblScanResult.ForeColor = Color.Red;
            }

            btnScanBank.Enabled = true;
            btnScanBank.Text = "Scan Bank";
        };

        AddSeparator(contentPanel, leftMargin, ref y, contentWidth);

        // ===== Step 3: Train PatchCore Models =====
        AddSectionHeader(contentPanel, "Step 3: Train PatchCore Models", leftMargin, ref y);

        // Model output directory
        var lblModelOutDir = new Label
        {
            Text = "Model Output Directory:",
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(leftMargin, y),
            Size = new Size(200, 20)
        };
        contentPanel.Controls.Add(lblModelOutDir);
        y += 22;

        var txtModelOutDir = new TextBox
        {
            PlaceholderText = @"e.g. E:\AIInspect\projects\label_check\models\20260301_120000",
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth - 90, 25),
            Name = "txtLabelModelOutDir"
        };
        contentPanel.Controls.Add(txtModelOutDir);

        var btnBrowseModelDir = new Button
        {
            Text = "Browse",
            Size = new Size(80, 26),
            Location = new Point(leftMargin + contentWidth - 82, y - 1),
            FlatStyle = FlatStyle.Flat
        };
        btnBrowseModelDir.Click += (s, e) =>
        {
            using var fbd = new FolderBrowserDialog { Description = "Select model output directory" };
            if (fbd.ShowDialog() == DialogResult.OK)
                txtModelOutDir.Text = fbd.SelectedPath;
        };
        contentPanel.Controls.Add(btnBrowseModelDir);
        y += 32;

        // Target project (optional, for auto-activate)
        var lblTargetProject = new Label
        {
            Text = "Target Project ID (for auto-activate, optional):",
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(leftMargin, y),
            Size = new Size(320, 20)
        };
        contentPanel.Controls.Add(lblTargetProject);
        y += 22;

        var txtTargetProject = new TextBox
        {
            PlaceholderText = "e.g. label_check",
            Location = new Point(leftMargin, y),
            Size = new Size(250, 25),
            Name = "txtLabelTargetProject"
        };
        contentPanel.Controls.Add(txtTargetProject);

        var chkAutoActivate = new CheckBox
        {
            Text = "Auto-activate after training",
            Checked = true,
            Location = new Point(leftMargin + 270, y + 2),
            Size = new Size(220, 22),
            Font = new Font("Segoe UI", 9.5f)
        };
        contentPanel.Controls.Add(chkAutoActivate);
        y += 32;

        // Training parameters row
        var lblParams = new Label
        {
            Text = "Training Parameters:",
            Font = new Font("Segoe UI", 9.5f, FontStyle.Bold),
            Location = new Point(leftMargin, y),
            Size = new Size(200, 20)
        };
        contentPanel.Controls.Add(lblParams);
        y += 24;

        // img_size
        var lblImgSize = new Label { Text = "img_size:", Location = new Point(leftMargin, y), Size = new Size(60, 20), Font = new Font("Segoe UI", 9) };
        var nudImgSize = new NumericUpDown { Minimum = 32, Maximum = 512, Value = 128, Location = new Point(leftMargin + 65, y - 2), Size = new Size(60, 25) };
        contentPanel.Controls.Add(lblImgSize);
        contentPanel.Controls.Add(nudImgSize);

        // max_patches
        var lblMaxPatch = new Label { Text = "max_patches:", Location = new Point(leftMargin + 140, y), Size = new Size(85, 20), Font = new Font("Segoe UI", 9) };
        var nudMaxPatch = new NumericUpDown { Minimum = 1000, Maximum = 100000, Value = 30000, Increment = 1000, Location = new Point(leftMargin + 230, y - 2), Size = new Size(80, 25) };
        contentPanel.Controls.Add(lblMaxPatch);
        contentPanel.Controls.Add(nudMaxPatch);

        // k
        var lblK = new Label { Text = "k:", Location = new Point(leftMargin + 330, y), Size = new Size(20, 20), Font = new Font("Segoe UI", 9) };
        var nudK = new NumericUpDown { Minimum = 1, Maximum = 20, Value = 1, Location = new Point(leftMargin + 350, y - 2), Size = new Size(50, 25) };
        contentPanel.Controls.Add(lblK);
        contentPanel.Controls.Add(nudK);

        // topk
        var lblTopK = new Label { Text = "topk:", Location = new Point(leftMargin + 415, y), Size = new Size(35, 20), Font = new Font("Segoe UI", 9) };
        var nudTopK = new NumericUpDown { Minimum = 1, Maximum = 100, Value = 10, Location = new Point(leftMargin + 455, y - 2), Size = new Size(55, 25) };
        contentPanel.Controls.Add(lblTopK);
        contentPanel.Controls.Add(nudTopK);

        // p_thr
        var lblPThr = new Label { Text = "p_thr:", Location = new Point(leftMargin + 525, y), Size = new Size(40, 20), Font = new Font("Segoe UI", 9) };
        var txtPThr = new TextBox { Text = "0.995", Location = new Point(leftMargin + 570, y - 2), Size = new Size(60, 25) };
        contentPanel.Controls.Add(lblPThr);
        contentPanel.Controls.Add(txtPThr);
        y += 32;

        // Start Training button
        var btnStartTrain = new Button
        {
            Text = "Start Training",
            Size = new Size(140, 36),
            Location = new Point(leftMargin, y),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(40, 167, 69),
            ForeColor = Color.White,
            Font = new Font("Segoe UI", 10, FontStyle.Bold)
        };
        contentPanel.Controls.Add(btnStartTrain);

        var lblTrainStatus = new Label
        {
            Text = "",
            Font = new Font("Segoe UI", 9.5f),
            Location = new Point(leftMargin + 150, y + 8),
            Size = new Size(contentWidth - 160, 22),
            Name = "lblLabelTrainStatus"
        };
        contentPanel.Controls.Add(lblTrainStatus);
        y += 42;

        // Progress bar
        var progressBar = new ProgressBar
        {
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 20),
            Minimum = 0,
            Maximum = 100,
            Value = 0,
            Name = "prgLabelTrain"
        };
        contentPanel.Controls.Add(progressBar);
        y += 28;

        // Training log
        var txtTrainLog = new TextBox
        {
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Both,
            WordWrap = false,
            Location = new Point(leftMargin, y),
            Size = new Size(contentWidth, 200),
            Font = new Font("Consolas", 8.5f),
            BackColor = Color.FromArgb(30, 30, 30),
            ForeColor = Color.FromArgb(220, 220, 220),
            Name = "txtLabelTrainLog"
        };
        contentPanel.Controls.Add(txtTrainLog);
        y += 208;

        // Start Training handler
        btnStartTrain.Click += async (s, e) =>
        {
            var bankDir = txtBankDir.Text.Trim();
            var modelOutDir = txtModelOutDir.Text.Trim();

            if (string.IsNullOrEmpty(bankDir))
            {
                MessageBox.Show("Please set the glyph bank directory in Step 1.", "Warning",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            if (string.IsNullOrEmpty(modelOutDir))
            {
                MessageBox.Show("Please set the model output directory.", "Warning",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            if (!double.TryParse(txtPThr.Text, out double pThr))
                pThr = 0.995;

            btnStartTrain.Enabled = false;
            btnStartTrain.Text = "Training...";
            lblTrainStatus.Text = "Starting training...";
            lblTrainStatus.ForeColor = Color.Black;
            progressBar.Value = 0;
            txtTrainLog.Clear();

            var resp = await _apiClient.LabelStartTrainingAsync(
                bankDir: bankDir,
                outputModelDir: modelOutDir,
                projectId: txtTargetProject.Text.Trim(),
                autoActivate: chkAutoActivate.Checked,
                imgSize: (int)nudImgSize.Value,
                maxPatchesPerClass: (int)nudMaxPatch.Value,
                k: (int)nudK.Value,
                scoreMode: "topk",
                topk: (int)nudTopK.Value,
                pThr: pThr
            );

            if (resp != null && resp.Ok)
            {
                var jobId = resp.JobId;

                // Poll until done
                while (true)
                {
                    await Task.Delay(2000);
                    var status = await _apiClient.LabelGetTrainStatusAsync(jobId);
                    if (status == null) break;

                    lblTrainStatus.Text = $"[{status.Progress:F0}%] {status.Message}";
                    progressBar.Value = Math.Min(100, (int)status.Progress);
                    txtTrainLog.Text = string.Join(Environment.NewLine, status.LogLines);
                    txtTrainLog.SelectionStart = txtTrainLog.Text.Length;
                    txtTrainLog.ScrollToCaret();

                    if (status.Status == "completed")
                    {
                        lblTrainStatus.Text = $"Training completed: {status.Message}";
                        lblTrainStatus.ForeColor = Color.Green;
                        progressBar.Value = 100;
                        break;
                    }
                    else if (status.Status == "failed")
                    {
                        lblTrainStatus.Text = $"Training FAILED: {status.Error ?? status.Message}";
                        lblTrainStatus.ForeColor = Color.Red;
                        break;
                    }
                }
            }
            else
            {
                lblTrainStatus.Text = "Failed to start training. Check service connection and paths.";
                lblTrainStatus.ForeColor = Color.Red;
            }

            btnStartTrain.Enabled = true;
            btnStartTrain.Text = "Start Training";
        };

        _tabControl.TabPages.Insert(0, tab);
        _tabControl.SelectedTab = tab;
    }

    private static void AddSectionHeader(Panel parent, string text, int x, ref int y)
    {
        var lbl = new Label
        {
            Text = text,
            Font = new Font("Segoe UI", 12, FontStyle.Bold),
            Location = new Point(x, y),
            Size = new Size(400, 26)
        };
        parent.Controls.Add(lbl);
        y += 30;
    }

    private static void AddSeparator(Panel parent, int x, ref int y, int width)
    {
        var sep = new Label
        {
            BorderStyle = BorderStyle.Fixed3D,
            Location = new Point(x, y),
            Size = new Size(width, 2)
        };
        parent.Controls.Add(sep);
        y += 12;
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
        foreach (var timer in _logTimers.Values)
        {
            timer.Stop();
            timer.Dispose();
        }
        _logTimers.Clear();
        _apiClient.Dispose();
        base.OnFormClosing(e);
    }
}
