# CricSense Metrics Module

A comprehensive metrics and analysis module for the CricSense cricket match summary system. This module provides detailed performance analysis, data quality assessment, and system evaluation capabilities.

## Features

### 📊 Data Quality Metrics
- **Data Completeness Score**: Measures the percentage of matches with complete data
- **File Integrity Score**: Evaluates the integrity of match files (both innings present)
- **Date Parsing Success Rate**: Tracks successful date extraction from match files
- **Missing Data Analysis**: Identifies matches with missing first/second innings

### 📈 Summary Quality Metrics
- **Generation Success Rate**: Percentage of successfully generated summaries
- **Score Accuracy Rate**: Validates accuracy of scores in generated summaries
- **Summary Consistency Score**: Measures consistency in summary format and structure
- **Average Summary Length**: Tracks the length of generated summaries

### ⚡ System Performance Metrics
- **Processing Time**: Average time to process each match
- **Memory Usage**: System memory consumption during processing
- **Error Rate**: Percentage of processing errors
- **File I/O Operations**: Number of file operations performed

### 🎯 Accuracy Metrics
- **Result Classification Accuracy**: Accuracy of match result predictions
- **Score Prediction Metrics**: MAE, RMSE, and R² for score predictions
- **Precision, Recall, F1-Score**: Classification performance metrics

## Installation

Install the required dependencies:

```bash
pip install -r requirements-metrics.txt
```

## Usage

### Basic Usage

```python
from metrics import CricSenseMetrics

# Initialize the analyzer
analyzer = CricSenseMetrics("sa20 data")

# Load data
analyzer.load_data()

# Generate comprehensive report
report = analyzer.generate_comprehensive_report()

# Print summary
analyzer.print_summary_report()

# Export detailed report
analyzer.export_metrics_report("metrics_report.json")
```

### Command Line Usage

```bash
# Run metrics analysis with console output
python metrics.py --print-report

# Run with custom data directory
python metrics.py --data-dir "path/to/data" --print-report

# Export to custom file
python metrics.py --output "my_metrics_report.json"
```

### Individual Metrics

```python
# Calculate specific metrics
data_quality = analyzer.calculate_data_quality_metrics()
summary_quality = analyzer.calculate_summary_quality_metrics()
system_performance = analyzer.calculate_system_performance_metrics()
accuracy_metrics = analyzer.calculate_accuracy_metrics()
```

## Metrics Explained

### Data Quality Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `data_completeness_score` | Percentage of matches with valid data | 0.0 - 1.0 |
| `file_integrity_score` | Percentage of matches with both innings | 0.0 - 1.0 |
| `date_parsing_success_rate` | Success rate of date extraction | 0.0 - 1.0 |
| `matches_with_both_innings` | Count of matches with complete data | Integer |

### Summary Quality Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `summary_generation_success_rate` | Success rate of summary generation | 0.0 - 1.0 |
| `score_accuracy_rate` | Accuracy of scores in summaries | 0.0 - 1.0 |
| `average_summary_length` | Average characters in summaries | Integer |
| `summary_consistency_score` | Consistency of summary format | 0.0 - 1.0 |

### System Performance Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `average_processing_time_per_match` | Time to process each match | Seconds |
| `total_processing_time` | Total time for all matches | Seconds |
| `memory_usage_mb` | Memory consumption | MB |
| `error_rate` | Percentage of processing errors | 0.0 - 1.0 |

### Accuracy Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `result_classification_accuracy` | Accuracy of result predictions | 0.0 - 1.0 |
| `score_prediction_mae` | Mean Absolute Error for scores | Float |
| `score_prediction_rmse` | Root Mean Square Error for scores | Float |
| `score_prediction_r2` | R-squared for score predictions | 0.0 - 1.0 |
| `result_f1_score` | F1-score for result classification | 0.0 - 1.0 |

## Report Structure

The comprehensive report includes:

```json
{
  "timestamp": "2025-10-23T12:52:44.136179",
  "data_directory": "sa20 data",
  "total_matches_analyzed": 99,
  "overall_score": 0.79,
  "data_quality": { ... },
  "summary_quality": { ... },
  "system_performance": { ... },
  "accuracy_metrics": { ... },
  "recommendations": [ ... ]
}
```

## Recommendations

The module automatically generates recommendations based on the analysis:

- **Data Quality Issues**: Suggestions for improving data completeness
- **Performance Optimization**: Recommendations for system performance improvements
- **Accuracy Enhancement**: Suggestions for improving prediction accuracy
- **Summary Quality**: Recommendations for better summary generation

## Example Output

```
================================================================================
CRICSENSE METRICS REPORT
================================================================================
Generated: 2025-10-23T12:52:44.136179
Data Directory: sa20 data
Total Matches Analyzed: 99
Overall Score: 0.79/1.0

DATA QUALITY:
  Completeness Score: 0.95
  File Integrity Score: 1.00
  Matches with Both Innings: 99

SUMMARY QUALITY:
  Generation Success Rate: 1.00
  Score Accuracy Rate: 1.00
  Average Summary Length: 1825 characters

SYSTEM PERFORMANCE:
  Average Processing Time: 0.02s
  Error Rate: 0.00
  Memory Usage: 0.00 MB

ACCURACY METRICS:
  Result Classification Accuracy: 1.00
  Score Prediction R²: 1.00
  F1 Score: 1.00

RECOMMENDATIONS:
  1. System performance is within acceptable parameters
================================================================================
```

## Dependencies

- `pandas>=1.3.0` - Data manipulation and analysis
- `numpy>=1.21.0` - Numerical computations
- `scikit-learn>=1.0.0` - Machine learning metrics
- `matplotlib>=3.5.0` - Plotting (optional)
- `seaborn>=0.11.0` - Statistical visualization (optional)
- `psutil>=5.8.0` - System and process utilities

## Integration with CricSense

The metrics module is designed to work seamlessly with the main CricSense system:

```python
# Import both modules
from cricsense_match_summary import get_match_files, load_match_data
from metrics import CricSenseMetrics

# Use together
matches = get_match_files("sa20 data")
analyzer = CricSenseMetrics("sa20 data")
report = analyzer.generate_comprehensive_report()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Directory Not Found**: Check the path to your data directory
3. **Memory Issues**: For large datasets, consider processing in batches

### Performance Tips

- Use SSD storage for better I/O performance
- Ensure sufficient RAM for large datasets
- Consider parallel processing for very large datasets

## Contributing

To extend the metrics module:

1. Add new metric calculation methods to the `CricSenseMetrics` class
2. Update the `_calculate_overall_score` method to include new metrics
3. Add new recommendations in the `_generate_recommendations` method
4. Update this documentation with new metrics

## License

This module is part of the CricSense project and follows the same license terms.
