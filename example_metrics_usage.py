#!/usr/bin/env python3
"""
Example usage of the CricSense Metrics Module

This script demonstrates how to use the metrics module to analyze
the performance of the CricSense cricket match summary system.
"""

from metrics import CricSenseMetrics
import json

def main():
    """Example usage of the CricSense metrics analyzer."""
    
    print("CricSense Metrics Analysis Example")
    print("=" * 50)
    
    # Initialize the metrics analyzer
    analyzer = CricSenseMetrics("sa20 data")
    
    # Load data
    print("Loading match data...")
    matches = analyzer.load_data()
    print(f"Loaded {len(matches)} matches")
    
    # Calculate individual metrics
    print("\nCalculating data quality metrics...")
    data_quality = analyzer.calculate_data_quality_metrics()
    print(f"Data completeness score: {data_quality['data_completeness_score']:.2f}")
    print(f"File integrity score: {data_quality['file_integrity_score']:.2f}")
    
    print("\nCalculating summary quality metrics...")
    summary_quality = analyzer.calculate_summary_quality_metrics()
    print(f"Summary generation success rate: {summary_quality['summary_generation_success_rate']:.2f}")
    print(f"Score accuracy rate: {summary_quality['score_accuracy_rate']:.2f}")
    
    print("\nCalculating system performance metrics...")
    system_performance = analyzer.calculate_system_performance_metrics()
    print(f"Average processing time: {system_performance['average_processing_time_per_match']:.2f}s")
    print(f"Error rate: {system_performance['error_rate']:.2f}")
    
    print("\nCalculating accuracy metrics...")
    accuracy_metrics = analyzer.calculate_accuracy_metrics()
    print(f"Result classification accuracy: {accuracy_metrics['result_classification_accuracy']:.2f}")
    print(f"Score prediction R²: {accuracy_metrics['score_prediction_r2']:.2f}")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report = analyzer.generate_comprehensive_report()
    
    # Print summary report
    analyzer.print_summary_report()
    
    # Export detailed report
    analyzer.export_metrics_report("detailed_metrics_report.json")
    print("\nDetailed report exported to 'detailed_metrics_report.json'")
    
    # Access specific metrics
    print(f"\nOverall system score: {report['overall_score']:.2f}/1.0")
    print(f"Total matches analyzed: {report['total_matches_analyzed']}")
    
    # Show recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

if __name__ == "__main__":
    main()





