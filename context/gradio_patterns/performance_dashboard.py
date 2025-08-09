# Gradio Performance Dashboard Patterns
# ====================================
#
# PRIORITY: ⭐⭐⭐⭐⭐ (Essential for OpenVINO GenAI monitoring)
#
# This file contains professional dashboard patterns for monitoring OpenVINO GenAI
# performance, metrics collection, and real-time visualization. These patterns are
# crucial for production deployments where performance monitoring is essential.
#
# Key Learning Points:
# - Real-time metrics collection and visualization
# - Professional dashboard layouts with multiple panels
# - Dynamic plot updates and data management
# - Performance monitoring best practices
# - Integration with ML model metrics

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import random
from datetime import datetime, timedelta
import json

# =======================================
# PATTERN 1: Basic Performance Monitor
# =======================================
# Source: demo/dashboard/run.py (adapted for OpenVINO GenAI)

def create_basic_performance_dashboard():
    """
    Basic performance monitoring dashboard for OpenVINO GenAI applications.
    Tracks essential metrics like response time, token throughput, and device utilization.
    """
    
    # Simulate OpenVINO GenAI metrics storage
    class OpenVINOMetrics:
        def __init__(self):
            self.response_times = []
            self.token_rates = []
            self.device_utilization = []
            self.error_counts = []
            self.timestamps = []
            
        def add_metric(self, response_time, tokens_per_sec, device_util, errors=0):
            now = datetime.now()
            self.response_times.append(response_time)
            self.token_rates.append(tokens_per_sec)
            self.device_utilization.append(device_util)
            self.error_counts.append(errors)
            self.timestamps.append(now)
            
            # Keep only last 100 measurements
            if len(self.timestamps) > 100:
                self.response_times.pop(0)
                self.token_rates.pop(0)
                self.device_utilization.pop(0)
                self.error_counts.pop(0)
                self.timestamps.pop(0)
    
    metrics = OpenVINOMetrics()
    
    def generate_response_time_plot():
        """Generate response time visualization"""
        if not metrics.timestamps:
            return gr.Plot(visible=False)
            
        df = pd.DataFrame({
            'timestamp': metrics.timestamps,
            'response_time': metrics.response_times
        })
        
        plot = px.line(df, x='timestamp', y='response_time',
                      title='OpenVINO GenAI Response Times',
                      labels={'response_time': 'Response Time (seconds)'})
        plot.update_layout(title_x=0.5)
        return gr.Plot(value=plot, visible=True)
    
    def generate_throughput_plot():
        """Generate token throughput visualization"""
        if not metrics.timestamps:
            return gr.Plot(visible=False)
            
        df = pd.DataFrame({
            'timestamp': metrics.timestamps,
            'tokens_per_sec': metrics.token_rates
        })
        
        plot = px.bar(df.tail(20), x='timestamp', y='tokens_per_sec',
                     title='Token Generation Throughput',
                     labels={'tokens_per_sec': 'Tokens/Second'})
        plot.update_layout(title_x=0.5)
        return gr.Plot(value=plot, visible=True)
    
    def generate_device_utilization_plot():
        """Generate device utilization visualization"""
        if not metrics.timestamps:
            return gr.Plot(visible=False)
            
        df = pd.DataFrame({
            'timestamp': metrics.timestamps,
            'utilization': metrics.device_utilization
        })
        
        plot = px.area(df, x='timestamp', y='utilization',
                      title='NPU/CPU Device Utilization',
                      labels={'utilization': 'Utilization %'})
        plot.update_layout(title_x=0.5, yaxis_range=[0, 100])
        return gr.Plot(value=plot, visible=True)
    
    def simulate_metrics_update():
        """Simulate adding new metrics (replace with real OpenVINO data)"""
        # Simulate realistic OpenVINO GenAI metrics
        response_time = random.uniform(0.5, 3.0)  # 0.5-3 seconds
        tokens_per_sec = random.uniform(15, 45)   # 15-45 tokens/second
        device_util = random.uniform(60, 95)      # 60-95% utilization
        errors = random.randint(0, 1)             # Occasional errors
        
        metrics.add_metric(response_time, tokens_per_sec, device_util, errors)
        
        return (
            generate_response_time_plot(),
            generate_throughput_plot(), 
            generate_device_utilization_plot(),
            get_summary_stats()
        )
    
    def get_summary_stats():
        """Get summary statistics"""
        if not metrics.response_times:
            return {"status": "No data available"}
            
        avg_response = sum(metrics.response_times) / len(metrics.response_times)
        avg_throughput = sum(metrics.token_rates) / len(metrics.token_rates)
        avg_utilization = sum(metrics.device_utilization) / len(metrics.device_utilization)
        total_errors = sum(metrics.error_counts)
        
        return {
            "total_requests": len(metrics.response_times),
            "avg_response_time": round(avg_response, 3),
            "avg_throughput": round(avg_throughput, 1),
            "avg_device_utilization": round(avg_utilization, 1),
            "total_errors": total_errors,
            "uptime": f"{len(metrics.timestamps) * 2} seconds"  # Assuming 2s intervals
        }
    
    with gr.Blocks(title="OpenVINO GenAI Performance Dashboard") as demo:
        gr.Markdown("# 🚀 OpenVINO GenAI Performance Dashboard")
        gr.Markdown("Real-time monitoring of NPU/CPU performance, token throughput, and system metrics")
        
        with gr.Row():
            update_btn = gr.Button("📊 Update Metrics", variant="primary")
            auto_refresh = gr.Checkbox(label="🔄 Auto-refresh (every 5s)", value=False)
        
        with gr.Row():
            with gr.Column():
                response_plot = gr.Plot(label="Response Times")
                device_plot = gr.Plot(label="Device Utilization") 
            with gr.Column():
                throughput_plot = gr.Plot(label="Token Throughput")
                summary_stats = gr.JSON(label="Summary Statistics")
        
        # Manual update
        update_btn.click(
            simulate_metrics_update,
            outputs=[response_plot, throughput_plot, device_plot, summary_stats]
        )
        
        # Auto-refresh setup (simplified - in practice use gr.Timer)
        def auto_update_handler():
            return simulate_metrics_update()
        
        # Initial load
        demo.load(
            simulate_metrics_update,
            outputs=[response_plot, throughput_plot, device_plot, summary_stats]
        )
    
    return demo

# =========================================
# PATTERN 2: Advanced Metrics Dashboard
# =========================================
# Comprehensive dashboard with multiple metrics and real-time updates

def create_advanced_metrics_dashboard():
    """
    Advanced dashboard with comprehensive OpenVINO GenAI metrics,
    model comparison, and performance analysis tools.
    """
    
    class AdvancedMetrics:
        def __init__(self):
            self.data = []
            
        def add_request(self, model_name, device, input_tokens, output_tokens, 
                       response_time, temperature, compilation_time=None):
            entry = {
                'timestamp': datetime.now(),
                'model': model_name,
                'device': device,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'response_time': response_time,
                'tokens_per_second': output_tokens / response_time if response_time > 0 else 0,
                'temperature': temperature,
                'compilation_time': compilation_time
            }
            self.data.append(entry)
            
            # Keep last 500 entries
            if len(self.data) > 500:
                self.data.pop(0)
    
    metrics = AdvancedMetrics()
    
    def create_performance_comparison_plot(models_to_compare, device_filter):
        """Create model performance comparison"""
        if not metrics.data:
            return gr.Plot(visible=False)
        
        df = pd.DataFrame(metrics.data)
        
        # Apply filters
        if device_filter != "All":
            df = df[df['device'] == device_filter]
        if models_to_compare:
            df = df[df['model'].isin(models_to_compare)]
        
        if df.empty:
            return gr.Plot(visible=False)
        
        # Group by model and calculate averages
        comparison = df.groupby('model').agg({
            'tokens_per_second': 'mean',
            'response_time': 'mean',
            'total_tokens': 'mean'
        }).reset_index()
        
        plot = px.scatter(comparison, x='response_time', y='tokens_per_second',
                         size='total_tokens', color='model',
                         title='Model Performance Comparison',
                         labels={
                             'response_time': 'Avg Response Time (s)',
                             'tokens_per_second': 'Avg Tokens/Second'
                         })
        plot.update_layout(title_x=0.5)
        return gr.Plot(value=plot, visible=True)
    
    def create_device_comparison_plot():
        """Create device performance comparison"""
        if not metrics.data:
            return gr.Plot(visible=False)
        
        df = pd.DataFrame(metrics.data)
        device_stats = df.groupby('device').agg({
            'tokens_per_second': ['mean', 'std'],
            'response_time': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        device_stats.columns = ['_'.join(col).strip() for col in device_stats.columns]
        device_stats = device_stats.reset_index()
        
        plot = px.bar(device_stats, x='device', y='tokens_per_second_mean',
                     error_y='tokens_per_second_std',
                     title='Device Performance Comparison',
                     labels={'tokens_per_second_mean': 'Average Tokens/Second'})
        plot.update_layout(title_x=0.5)
        return gr.Plot(value=plot, visible=True)
    
    def create_timeline_analysis():
        """Create timeline performance analysis"""
        if not metrics.data:
            return gr.Plot(visible=False)
        
        df = pd.DataFrame(metrics.data)
        df['hour'] = df['timestamp'].dt.floor('H')
        
        hourly_stats = df.groupby(['hour', 'device']).agg({
            'tokens_per_second': 'mean',
            'response_time': 'mean'
        }).reset_index()
        
        plot = px.line(hourly_stats, x='hour', y='tokens_per_second',
                      color='device', title='Performance Over Time',
                      labels={'tokens_per_second': 'Tokens/Second'})
        plot.update_layout(title_x=0.5)
        return gr.Plot(value=plot, visible=True)
    
    def generate_performance_report():
        """Generate comprehensive performance report"""
        if not metrics.data:
            return "No data available for report generation."
        
        df = pd.DataFrame(metrics.data)
        
        report = {
            "summary": {
                "total_requests": len(df),
                "avg_tokens_per_second": round(df['tokens_per_second'].mean(), 2),
                "avg_response_time": round(df['response_time'].mean(), 2),
                "total_tokens_processed": int(df['total_tokens'].sum())
            },
            "by_device": df.groupby('device').agg({
                'tokens_per_second': ['mean', 'max', 'min'],
                'response_time': ['mean', 'max', 'min']
            }).round(2).to_dict(),
            "by_model": df.groupby('model').agg({
                'tokens_per_second': 'mean',
                'response_time': 'mean'
            }).round(2).to_dict(),
            "recent_performance": df.tail(10)[['timestamp', 'model', 'device', 
                                           'tokens_per_second', 'response_time']].to_dict('records')
        }
        
        return json.dumps(report, indent=2, default=str)
    
    def simulate_advanced_metrics():
        """Simulate adding advanced metrics"""
        models = ["qwen3-8b-int4", "llama2-7b-int8", "mistral-7b-fp16"]
        devices = ["NPU", "CPU", "GPU"]
        
        for _ in range(5):  # Add 5 random entries
            model = random.choice(models)
            device = random.choice(devices)
            input_tokens = random.randint(50, 500)
            output_tokens = random.randint(100, 300)
            response_time = random.uniform(1.0, 5.0)
            temperature = random.uniform(0.3, 1.2)
            
            # NPU has compilation overhead sometimes
            compilation_time = random.uniform(2.0, 8.0) if device == "NPU" and random.random() < 0.3 else None
            
            metrics.add_request(model, device, input_tokens, output_tokens,
                              response_time, temperature, compilation_time)
        
        return "✅ Added 5 new metric entries"
    
    with gr.Blocks(title="Advanced OpenVINO GenAI Analytics") as demo:
        gr.Markdown("# 📊 Advanced OpenVINO GenAI Performance Analytics")
        gr.Markdown("Comprehensive monitoring, comparison, and analysis of OpenVINO GenAI models")
        
        with gr.Row():
            with gr.Column(scale=2):
                model_selector = gr.CheckboxGroup(
                    choices=["qwen3-8b-int4", "llama2-7b-int8", "mistral-7b-fp16"],
                    label="Models to Compare",
                    value=["qwen3-8b-int4"]
                )
            with gr.Column(scale=1):
                device_filter = gr.Dropdown(
                    choices=["All", "NPU", "CPU", "GPU"],
                    value="All",
                    label="Device Filter"
                )
            with gr.Column(scale=1):
                simulate_btn = gr.Button("🎲 Add Sample Data", variant="secondary")
                update_btn = gr.Button("📊 Update Dashboard", variant="primary")
        
        with gr.Tabs():
            with gr.Tab("Performance Comparison"):
                comparison_plot = gr.Plot(label="Model Performance Comparison")
                device_comparison_plot = gr.Plot(label="Device Comparison")
            
            with gr.Tab("Timeline Analysis"):
                timeline_plot = gr.Plot(label="Performance Over Time")
            
            with gr.Tab("Detailed Report"):
                report_output = gr.Code(label="Performance Report (JSON)", language="json")
                generate_report_btn = gr.Button("📋 Generate Report")
        
        # Event handlers
        simulate_btn.click(simulate_advanced_metrics, outputs=gr.Textbox(visible=False))
        
        update_btn.click(
            lambda models, device: (
                create_performance_comparison_plot(models, device),
                create_device_comparison_plot(),
                create_timeline_analysis()
            ),
            inputs=[model_selector, device_filter],
            outputs=[comparison_plot, device_comparison_plot, timeline_plot]
        )
        
        generate_report_btn.click(generate_performance_report, outputs=report_output)
        
        # Initial load with sample data
        demo.load(
            lambda: simulate_advanced_metrics() + update_btn.click(),
            outputs=gr.Textbox(visible=False)
        )
    
    return demo

# =========================================
# PATTERN 3: Real-Time Monitoring System  
# =========================================
# Live monitoring system with alerts and thresholds

def create_realtime_monitoring_system():
    """
    Real-time monitoring system with alerts and performance thresholds.
    Essential for production OpenVINO GenAI deployments.
    """
    
    class MonitoringSystem:
        def __init__(self):
            self.alerts = []
            self.thresholds = {
                'max_response_time': 5.0,
                'min_tokens_per_second': 10.0,
                'max_error_rate': 0.05,
                'max_memory_usage': 85.0
            }
            self.current_metrics = {}
            
        def check_alerts(self, metrics):
            alerts = []
            
            if metrics.get('response_time', 0) > self.thresholds['max_response_time']:
                alerts.append(f"⚠️ High response time: {metrics['response_time']:.2f}s")
            
            if metrics.get('tokens_per_second', 0) < self.thresholds['min_tokens_per_second']:
                alerts.append(f"⚠️ Low throughput: {metrics['tokens_per_second']:.1f} tok/s")
            
            if metrics.get('error_rate', 0) > self.thresholds['max_error_rate']:
                alerts.append(f"🚨 High error rate: {metrics['error_rate']:.2%}")
            
            return alerts
    
    monitor = MonitoringSystem()
    
    def update_realtime_metrics():
        """Update real-time metrics with alert checking"""
        # Simulate current system state
        current = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'response_time': random.uniform(0.8, 6.0),
            'tokens_per_second': random.uniform(8, 35),
            'active_requests': random.randint(0, 10),
            'queue_length': random.randint(0, 5),
            'memory_usage': random.uniform(60, 95),
            'error_rate': random.uniform(0, 0.08),
            'device_temperature': random.uniform(45, 75)
        }
        
        monitor.current_metrics = current
        alerts = monitor.check_alerts(current)
        
        # Create status indicator
        status = "🟢 Healthy"
        if alerts:
            if any("🚨" in alert for alert in alerts):
                status = "🔴 Critical"
            else:
                status = "🟡 Warning"
        
        # Format metrics for display
        metrics_display = {
            "System Status": status,
            "Response Time": f"{current['response_time']:.2f}s",
            "Token Throughput": f"{current['tokens_per_second']:.1f} tok/s",
            "Active Requests": current['active_requests'],
            "Queue Length": current['queue_length'],
            "Memory Usage": f"{current['memory_usage']:.1f}%",
            "Error Rate": f"{current['error_rate']:.2%}",
            "Device Temperature": f"{current['device_temperature']:.1f}°C"
        }
        
        alerts_text = "\n".join(alerts) if alerts else "✅ No alerts"
        
        return metrics_display, alerts_text, status
    
    def update_threshold(threshold_name, new_value):
        """Update monitoring thresholds"""
        if threshold_name in monitor.thresholds:
            monitor.thresholds[threshold_name] = new_value
            return f"✅ Updated {threshold_name} threshold to {new_value}"
        return f"❌ Unknown threshold: {threshold_name}"
    
    with gr.Blocks(title="OpenVINO GenAI Real-Time Monitor") as demo:
        gr.Markdown("# 🔴 OpenVINO GenAI Real-Time Monitoring System")
        gr.Markdown("Live performance monitoring with automated alerts and threshold management")
        
        with gr.Row():
            with gr.Column(scale=2):
                status_display = gr.HTML("<h2>🟢 System Status: Initializing...</h2>")
                metrics_json = gr.JSON(label="Current Metrics")
            with gr.Column(scale=1):
                alerts_display = gr.Textbox(
                    label="🚨 Active Alerts",
                    lines=5,
                    placeholder="No alerts"
                )
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Metrics", variant="primary")
            auto_refresh_checkbox = gr.Checkbox(label="Auto-refresh (5s intervals)", value=True)
        
        with gr.Accordion("⚙️ Alert Thresholds", open=False):
            with gr.Row():
                with gr.Column():
                    max_response_threshold = gr.Slider(
                        minimum=1.0, maximum=10.0, value=5.0, step=0.1,
                        label="Max Response Time (seconds)"
                    )
                    min_throughput_threshold = gr.Slider(
                        minimum=5.0, maximum=50.0, value=10.0, step=1.0,
                        label="Min Tokens/Second"
                    )
                with gr.Column():
                    max_error_threshold = gr.Slider(
                        minimum=0.01, maximum=0.20, value=0.05, step=0.01,
                        label="Max Error Rate"
                    )
                    max_memory_threshold = gr.Slider(
                        minimum=70.0, maximum=95.0, value=85.0, step=1.0,
                        label="Max Memory Usage (%)"
                    )
            
            update_thresholds_btn = gr.Button("💾 Update Thresholds")
        
        # Event handlers
        def refresh_with_status():
            metrics, alerts, status = update_realtime_metrics()
            status_html = f"<h2>{status}</h2>"
            return metrics, alerts, status_html
        
        refresh_btn.click(
            refresh_with_status,
            outputs=[metrics_json, alerts_display, status_display]
        )
        
        def update_all_thresholds(max_response, min_throughput, max_error, max_memory):
            monitor.thresholds['max_response_time'] = max_response
            monitor.thresholds['min_tokens_per_second'] = min_throughput  
            monitor.thresholds['max_error_rate'] = max_error
            monitor.thresholds['max_memory_usage'] = max_memory
            return "✅ All thresholds updated successfully"
        
        update_thresholds_btn.click(
            update_all_thresholds,
            inputs=[max_response_threshold, min_throughput_threshold, 
                   max_error_threshold, max_memory_threshold],
            outputs=gr.Textbox(label="Update Status", visible=True)
        )
        
        # Initial load
        demo.load(refresh_with_status, outputs=[metrics_json, alerts_display, status_display])
    
    return demo

# =====================================
# INTEGRATION GUIDELINES FOR OPENVINO
# =====================================

"""
OPENVINO GENAI INTEGRATION GUIDELINES:
=====================================

1. METRICS COLLECTION:
   ```python
   # In your OpenVINO GenAI application:
   class OpenVINOMetricsCollector:
       def track_request(self, start_time, end_time, input_tokens, output_tokens, device, errors=0):
           response_time = end_time - start_time
           tokens_per_sec = output_tokens / response_time if response_time > 0 else 0
           
           # Store in your metrics system
           self.add_metric(response_time, tokens_per_sec, device, input_tokens, output_tokens, errors)
   ```

2. REAL-TIME MONITORING:
   ```python
   # Integrate with your streaming generation:
   def openvino_generate_with_monitoring(message, history):
       start_time = time.time()
       
       try:
           # Your OpenVINO generation
           streamer = YourStreamer(tokenizer)
           pipe.generate(prompt, config, streamer)
           
           for chunk in streamer:
               yield chunk
           
           # Record successful metrics
           end_time = time.time()
           collector.track_request(start_time, end_time, input_tokens, output_tokens, device)
           
       except Exception as e:
           # Record error metrics
           end_time = time.time()
           collector.track_request(start_time, end_time, input_tokens, 0, device, errors=1)
           raise
   ```

3. DEVICE-SPECIFIC MONITORING:
   ```python
   # NPU-specific monitoring
   def monitor_npu_performance():
       return {
           'compilation_time': get_npu_compilation_time(),
           'inference_time': get_npu_inference_time(),
           'memory_usage': get_npu_memory_usage(),
           'temperature': get_npu_temperature()
       }
   ```

RECOMMENDED USAGE:
=================

1. Use PATTERN 1 for basic performance monitoring during development
2. Use PATTERN 2 for comprehensive model comparison and analysis
3. Use PATTERN 3 for production monitoring with alerts

The real-time monitoring system (PATTERN 3) is essential for production
OpenVINO GenAI deployments where uptime and performance are critical.

DASHBOARD DEPLOYMENT:
====================

Deploy alongside your main application:
```python
# Main chat interface
chat_demo = your_openvino_chat_interface()

# Performance dashboard  
dashboard_demo = create_advanced_metrics_dashboard()

# Combine or deploy separately
gr.TabbedInterface([chat_demo, dashboard_demo], 
                  ["💬 Chat", "📊 Dashboard"]).launch()
```
"""

# Example usage:
if __name__ == "__main__":
    # Choose the dashboard pattern most suitable for your needs
    demo = create_advanced_metrics_dashboard()  # Recommended for production
    demo.launch()