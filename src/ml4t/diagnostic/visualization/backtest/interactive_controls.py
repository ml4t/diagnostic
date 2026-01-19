"""Interactive controls for backtest tearsheets.

Provides JavaScript-based UI components for interactive tearsheets:
- Date range selector with presets (1M, 3M, YTD, 1Y, All)
- Metric filter dropdown
- Section navigation
- Drill-down functionality

These controls enhance the HTML tearsheet with client-side interactivity.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# =============================================================================
# Date Range Selector
# =============================================================================

DATE_RANGE_PRESETS = {
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "YTD": "ytd",
    "1Y": 365,
    "3Y": 1095,
    "5Y": 1825,
    "All": None,
}


def get_date_range_html(
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    presets: list[str] | None = None,
    default_preset: str = "All",
    on_change_callback: str = "onDateRangeChange",
) -> str:
    """Generate HTML/JS for a date range selector with presets.

    Parameters
    ----------
    start_date : date or str, optional
        Minimum selectable date (data start)
    end_date : date or str, optional
        Maximum selectable date (data end)
    presets : list[str], optional
        Preset buttons to show (default: ["1M", "3M", "YTD", "1Y", "All"])
    default_preset : str
        Initially selected preset
    on_change_callback : str
        JavaScript function name to call on date change

    Returns
    -------
    str
        HTML string with date range selector
    """
    if presets is None:
        presets = ["1M", "3M", "YTD", "1Y", "All"]

    # Convert dates to strings
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()

    # Build preset buttons
    preset_buttons = []
    for preset in presets:
        active = "active" if preset == default_preset else ""
        preset_buttons.append(
            f'<button class="date-preset-btn {active}" data-preset="{preset}">{preset}</button>'
        )

    return f"""
    <div class="date-range-selector">
        <div class="preset-buttons">
            {"".join(preset_buttons)}
        </div>
        <div class="custom-range">
            <input type="date" id="start-date" value="{start_date or ""}" min="{start_date or ""}" max="{end_date or ""}">
            <span>to</span>
            <input type="date" id="end-date" value="{end_date or ""}" min="{start_date or ""}" max="{end_date or ""}">
        </div>
    </div>

    <style>
        .date-range-selector {{
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 10px 0;
            margin-bottom: 20px;
        }}
        .preset-buttons {{
            display: flex;
            gap: 5px;
        }}
        .date-preset-btn {{
            padding: 6px 12px;
            border: 1px solid #ddd;
            background: #f8f9fa;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }}
        .date-preset-btn:hover {{
            background: #e9ecef;
        }}
        .date-preset-btn.active {{
            background: #636EFA;
            color: white;
            border-color: #636EFA;
        }}
        .custom-range {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .custom-range input {{
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 13px;
        }}
    </style>

    <script>
        (function() {{
            const startDate = '{start_date or ""}';
            const endDate = '{end_date or ""}';

            document.querySelectorAll('.date-preset-btn').forEach(btn => {{
                btn.addEventListener('click', function() {{
                    // Update active state
                    document.querySelectorAll('.date-preset-btn').forEach(b => b.classList.remove('active'));
                    this.classList.add('active');

                    const preset = this.dataset.preset;
                    let newStart, newEnd = endDate;

                    if (preset === 'All') {{
                        newStart = startDate;
                    }} else if (preset === 'YTD') {{
                        const now = new Date(endDate);
                        newStart = new Date(now.getFullYear(), 0, 1).toISOString().split('T')[0];
                    }} else {{
                        const days = {{"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 1095, "5Y": 1825}}[preset];
                        if (days) {{
                            const end = new Date(endDate);
                            newStart = new Date(end - days * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
                        }}
                    }}

                    document.getElementById('start-date').value = newStart;
                    document.getElementById('end-date').value = newEnd;

                    if (typeof {on_change_callback} === 'function') {{
                        {on_change_callback}(newStart, newEnd);
                    }}
                }});
            }});

            // Custom date inputs
            ['start-date', 'end-date'].forEach(id => {{
                document.getElementById(id).addEventListener('change', function() {{
                    const newStart = document.getElementById('start-date').value;
                    const newEnd = document.getElementById('end-date').value;

                    // Clear preset active state
                    document.querySelectorAll('.date-preset-btn').forEach(b => b.classList.remove('active'));

                    if (typeof {on_change_callback} === 'function') {{
                        {on_change_callback}(newStart, newEnd);
                    }}
                }});
            }});
        }})();
    </script>
    """


# =============================================================================
# Metric Filter Dropdown
# =============================================================================


def get_metric_filter_html(
    metrics: list[str],
    default_metric: str | None = None,
    multi_select: bool = False,
    on_change_callback: str = "onMetricFilterChange",
) -> str:
    """Generate HTML/JS for a metric filter dropdown.

    Parameters
    ----------
    metrics : list[str]
        Available metric names
    default_metric : str, optional
        Initially selected metric
    multi_select : bool
        Whether to allow multiple selection
    on_change_callback : str
        JavaScript function name to call on change

    Returns
    -------
    str
        HTML string with metric filter dropdown
    """
    if default_metric is None and metrics:
        default_metric = metrics[0]

    options = []
    for metric in metrics:
        selected = "selected" if metric == default_metric else ""
        options.append(f'<option value="{metric}" {selected}>{metric}</option>')

    multiple_attr = "multiple" if multi_select else ""

    return f"""
    <div class="metric-filter">
        <label for="metric-select">Metric:</label>
        <select id="metric-select" {multiple_attr}>
            {"".join(options)}
        </select>
    </div>

    <style>
        .metric-filter {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .metric-filter label {{
            font-weight: 500;
            font-size: 14px;
        }}
        .metric-filter select {{
            padding: 6px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            min-width: 150px;
        }}
    </style>

    <script>
        document.getElementById('metric-select').addEventListener('change', function() {{
            const isMulti = {"true" if multi_select else "false"};
            const selected = isMulti
                ? Array.from(this.selectedOptions).map(o => o.value)
                : this.value;
            if (typeof {on_change_callback} === 'function') {{
                {on_change_callback}(selected);
            }}
        }});
    </script>
    """


# =============================================================================
# Section Navigation
# =============================================================================


def get_section_navigation_html(
    sections: list[dict[str, str]],
    sticky: bool = True,
) -> str:
    """Generate HTML/JS for section navigation sidebar.

    Parameters
    ----------
    sections : list[dict]
        List of {"id": "section-id", "title": "Section Title"}
    sticky : bool
        Whether navigation should stick to viewport

    Returns
    -------
    str
        HTML string with section navigation
    """
    nav_items = []
    for section in sections:
        nav_items.append(f'<a href="#{section["id"]}" class="nav-item">{section["title"]}</a>')

    position_style = "position: sticky; top: 20px;" if sticky else ""

    return f"""
    <nav class="section-nav" style="{position_style}">
        <div class="nav-title">Contents</div>
        {"".join(nav_items)}
    </nav>

    <style>
        .section-nav {{
            width: 200px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        .nav-title {{
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 12px;
            color: #495057;
        }}
        .nav-item {{
            display: block;
            padding: 6px 10px;
            margin: 2px 0;
            color: #666;
            text-decoration: none;
            font-size: 13px;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        .nav-item:hover {{
            background: #e9ecef;
            color: #333;
        }}
        .nav-item.active {{
            background: #636EFA;
            color: white;
        }}
    </style>

    <script>
        // Highlight current section in navigation
        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    document.querySelectorAll('.nav-item').forEach(item => {{
                        item.classList.remove('active');
                        if (item.getAttribute('href') === '#' + entry.target.id) {{
                            item.classList.add('active');
                        }}
                    }});
                }}
            }});
        }}, {{ threshold: 0.3 }});

        document.querySelectorAll('section').forEach(section => {{
            observer.observe(section);
        }});
    </script>
    """


# =============================================================================
# Drill-Down Modal
# =============================================================================


def get_drill_down_modal_html() -> str:
    """Generate HTML/JS for drill-down modal functionality.

    Returns
    -------
    str
        HTML string with modal component
    """
    return """
    <div id="drill-down-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modal-title">Details</h3>
                <span class="modal-close">&times;</span>
            </div>
            <div id="modal-body" class="modal-body">
                <!-- Dynamic content loaded here -->
            </div>
        </div>
    </div>

    <style>
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }
        .modal-content {
            background-color: #fff;
            margin: 5% auto;
            padding: 0;
            border-radius: 8px;
            width: 80%;
            max-width: 900px;
            max-height: 80vh;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            border-bottom: 1px solid #dee2e6;
            background: #f8f9fa;
        }
        .modal-header h3 {
            margin: 0;
            font-size: 18px;
        }
        .modal-close {
            font-size: 28px;
            font-weight: bold;
            color: #666;
            cursor: pointer;
        }
        .modal-close:hover {
            color: #333;
        }
        .modal-body {
            padding: 20px;
            overflow-y: auto;
            max-height: calc(80vh - 60px);
        }
    </style>

    <script>
        const modal = document.getElementById('drill-down-modal');
        const modalTitle = document.getElementById('modal-title');
        const modalBody = document.getElementById('modal-body');
        const closeBtn = document.querySelector('.modal-close');

        function showDrillDown(title, content) {
            modalTitle.textContent = title;
            modalBody.innerHTML = content;
            modal.style.display = 'block';
        }

        function hideDrillDown() {
            modal.style.display = 'none';
        }

        closeBtn.onclick = hideDrillDown;

        window.onclick = function(event) {
            if (event.target === modal) {
                hideDrillDown();
            }
        };

        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                hideDrillDown();
            }
        });
    </script>
    """


# =============================================================================
# Complete Interactive Toolbar
# =============================================================================


def get_interactive_toolbar_html(
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    metrics: list[str] | None = None,
    show_date_range: bool = True,
    show_metric_filter: bool = True,
    show_export_button: bool = True,
) -> str:
    """Generate a complete interactive toolbar for the tearsheet.

    Parameters
    ----------
    start_date : date or str, optional
        Data start date
    end_date : date or str, optional
        Data end date
    metrics : list[str], optional
        Available metrics for filtering
    show_date_range : bool
        Whether to show date range selector
    show_metric_filter : bool
        Whether to show metric filter
    show_export_button : bool
        Whether to show export button

    Returns
    -------
    str
        HTML string with complete toolbar
    """
    components = []

    if show_date_range:
        components.append(get_date_range_html(start_date, end_date))

    if show_metric_filter and metrics:
        components.append(get_metric_filter_html(metrics))

    export_btn = ""
    if show_export_button:
        export_btn = """
        <button class="export-btn" onclick="window.print()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Export
        </button>
        """

    return f"""
    <div class="tearsheet-toolbar">
        <div class="toolbar-left">
            {"".join(components)}
        </div>
        <div class="toolbar-right">
            {export_btn}
        </div>
    </div>

    <style>
        .tearsheet-toolbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            margin-bottom: 20px;
            border-bottom: 1px solid #dee2e6;
        }}
        .toolbar-left {{
            display: flex;
            gap: 30px;
            align-items: center;
        }}
        .toolbar-right {{
            display: flex;
            gap: 10px;
        }}
        .export-btn {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            background: #636EFA;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}
        .export-btn:hover {{
            background: #5258d9;
        }}

        @media print {{
            .tearsheet-toolbar {{
                display: none;
            }}
        }}
    </style>
    """


# =============================================================================
# Theme Switcher
# =============================================================================


def get_theme_switcher_html(
    themes: list[str] | None = None,
    default_theme: str = "default",
) -> str:
    """Generate HTML/JS for theme switcher.

    Parameters
    ----------
    themes : list[str], optional
        Available themes (default: ["default", "dark", "print"])
    default_theme : str
        Initially selected theme

    Returns
    -------
    str
        HTML string with theme switcher
    """
    if themes is None:
        themes = ["default", "dark", "print"]

    theme_labels = {
        "default": "Light",
        "dark": "Dark",
        "print": "Print",
        "presentation": "Present",
    }

    buttons = []
    for theme in themes:
        active = "active" if theme == default_theme else ""
        label = theme_labels.get(theme, theme.title())
        buttons.append(f'<button class="theme-btn {active}" data-theme="{theme}">{label}</button>')

    return f"""
    <div class="theme-switcher">
        {"".join(buttons)}
    </div>

    <style>
        .theme-switcher {{
            display: flex;
            gap: 5px;
        }}
        .theme-btn {{
            padding: 5px 12px;
            border: 1px solid #ddd;
            background: #fff;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}
        .theme-btn:hover {{
            background: #f0f0f0;
        }}
        .theme-btn.active {{
            background: #636EFA;
            color: white;
            border-color: #636EFA;
        }}
    </style>

    <script>
        document.querySelectorAll('.theme-btn').forEach(btn => {{
            btn.addEventListener('click', function() {{
                document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');

                const theme = this.dataset.theme;
                document.documentElement.setAttribute('data-theme', theme === 'dark' ? 'dark' : 'light');

                // Notify Plotly charts to update
                if (typeof Plotly !== 'undefined') {{
                    document.querySelectorAll('.js-plotly-plot').forEach(plot => {{
                        const bgColor = theme === 'dark' ? '#1E1E1E' : '#FFFFFF';
                        const textColor = theme === 'dark' ? '#E0E0E0' : '#2E2E2E';
                        Plotly.relayout(plot, {{
                            'paper_bgcolor': bgColor,
                            'plot_bgcolor': bgColor,
                            'font.color': textColor,
                        }});
                    }});
                }}
            }});
        }});
    </script>
    """
