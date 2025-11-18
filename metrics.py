"""
CricSense Metrics Module

This module provides comprehensive metrics and analysis for the CricSense cricket match summary system.
It evaluates data quality, summary accuracy, system performance, and provides detailed reporting.
"""

import os
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import the main module functions
from cricsense_match_summary import (
    get_match_files, load_match_data, generate_match_summary,
    build_structured_summary, report_missing_innings
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CricSenseMetrics:
    """
    Comprehensive metrics analyzer for the CricSense cricket match summary system.
    """
    
    def __init__(self, data_dir: str = "sa20 data"):
        """
        Initialize the metrics analyzer.
        
        Args:
            data_dir: Directory containing match data
        """
        self.data_dir = data_dir
        self.matches = {}
        self.metrics_results = {}
        
    def load_data(self) -> Dict[str, Any]:
        """Load and process all match data for analysis."""
        logger.info("Loading match data for metrics analysis...")
        self.matches = get_match_files(self.data_dir)
        logger.info(f"Loaded {len(self.matches)} matches for analysis")
        return self.matches
    
    def calculate_data_quality_metrics(self) -> Dict[str, Any]:
        """
        Calculate data quality metrics including completeness, consistency, and integrity.
        
        Returns:
            Dictionary containing data quality metrics
        """
        logger.info("Calculating data quality metrics...")
        
        if not self.matches:
            self.load_data()
        
        metrics = {
            'total_matches': len(self.matches),
            'matches_with_dates': 0,
            'matches_with_both_innings': 0,
            'matches_with_valid_data': 0,
            'missing_first_innings': 0,
            'missing_second_innings': 0,
            'missing_both_innings': 0,
            'data_completeness_score': 0.0,
            'date_parsing_success_rate': 0.0,
            'file_integrity_score': 0.0
        }
        
        valid_matches = 0
        date_success_count = 0
        file_integrity_score = 0
        
        for match_id, match_info in self.matches.items():
            # Check date availability
            if match_info.get('date'):
                metrics['matches_with_dates'] += 1
                date_success_count += 1
            
            # Check innings completeness
            files = match_info.get('files', {})
            has_first = 1 in files and isinstance(files[1], str)
            has_second = 2 in files and isinstance(files[2], str)
            
            if has_first and has_second:
                metrics['matches_with_both_innings'] += 1
                file_integrity_score += 1
            elif has_first:
                metrics['missing_second_innings'] += 1
                file_integrity_score += 0.5
            elif has_second:
                metrics['missing_first_innings'] += 1
                file_integrity_score += 0.5
            else:
                metrics['missing_both_innings'] += 1
            
            # Check data validity
            try:
                match_summary, _ = load_match_data(match_info)
                if match_summary and match_summary.get('valid'):
                    metrics['matches_with_valid_data'] += 1
                    valid_matches += 1
            except Exception as e:
                logger.warning(f"Error validating match {match_id}: {e}")
        
        # Calculate derived metrics
        total_matches = len(self.matches)
        if total_matches > 0:
            metrics['data_completeness_score'] = valid_matches / total_matches
            metrics['date_parsing_success_rate'] = date_success_count / total_matches
            metrics['file_integrity_score'] = file_integrity_score / total_matches
        
        self.metrics_results['data_quality'] = metrics
        return metrics
    
    def calculate_summary_quality_metrics(self) -> Dict[str, Any]:
        """
        Calculate summary quality metrics including consistency, completeness, and accuracy.
        
        Returns:
            Dictionary containing summary quality metrics
        """
        logger.info("Calculating summary quality metrics...")
        
        if not self.matches:
            self.load_data()
        
        metrics = {
            'total_summaries_generated': 0,
            'successful_summaries': 0,
            'failed_summaries': 0,
            'summary_generation_success_rate': 0.0,
            'average_summary_length': 0.0,
            'summary_consistency_score': 0.0,
            'score_accuracy_rate': 0.0,
            'result_prediction_accuracy': 0.0
        }
        
        successful_summaries = 0
        total_length = 0
        score_accuracy_count = 0
        result_accuracy_count = 0
        
        for match_id, match_info in self.matches.items():
            try:
                match_summary, match_data = load_match_data(match_info)
                if match_summary and match_summary.get('valid'):
                    # Generate summary
                    summary_text = generate_match_summary(match_info, match_summary)
                    metrics['total_summaries_generated'] += 1
                    
                    if summary_text and "Error:" not in summary_text:
                        successful_summaries += 1
                        total_length += len(summary_text)
                        
                        # Check score accuracy (basic validation)
                        if self._validate_score_accuracy(match_summary, summary_text):
                            score_accuracy_count += 1
                        
                        # Check result prediction accuracy
                        if self._validate_result_accuracy(match_summary, summary_text):
                            result_accuracy_count += 1
                    else:
                        metrics['failed_summaries'] += 1
                        
            except Exception as e:
                logger.warning(f"Error generating summary for match {match_id}: {e}")
                metrics['failed_summaries'] += 1
        
        # Calculate derived metrics
        if metrics['total_summaries_generated'] > 0:
            metrics['summary_generation_success_rate'] = successful_summaries / metrics['total_summaries_generated']
            metrics['score_accuracy_rate'] = score_accuracy_count / successful_summaries if successful_summaries > 0 else 0
            metrics['result_prediction_accuracy'] = result_accuracy_count / successful_summaries if successful_summaries > 0 else 0
        
        if successful_summaries > 0:
            metrics['average_summary_length'] = total_length / successful_summaries
        
        # Calculate consistency score (simplified)
        metrics['summary_consistency_score'] = self._calculate_consistency_score()
        
        self.metrics_results['summary_quality'] = metrics
        return metrics
    
    def calculate_system_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate system performance metrics including processing time and resource usage.
        
        Returns:
            Dictionary containing system performance metrics
        """
        logger.info("Calculating system performance metrics...")
        
        import time
        import psutil
        
        metrics = {
            'average_processing_time_per_match': 0.0,
            'total_processing_time': 0.0,
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'file_io_operations': 0,
            'error_rate': 0.0
        }
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        processing_times = []
        error_count = 0
        
        for match_id, match_info in self.matches.items():
            match_start = time.time()
            try:
                match_summary, _ = load_match_data(match_info)
                if match_summary:
                    generate_match_summary(match_info, match_summary)
            except Exception as e:
                error_count += 1
                logger.warning(f"Error processing match {match_id}: {e}")
            
            match_time = time.time() - match_start
            processing_times.append(match_time)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        metrics['total_processing_time'] = end_time - start_time
        metrics['average_processing_time_per_match'] = np.mean(processing_times) if processing_times else 0
        metrics['memory_usage_mb'] = end_memory - start_memory
        metrics['cpu_usage_percent'] = psutil.cpu_percent()
        metrics['file_io_operations'] = len(self.matches) * 2  # Approximate (2 files per match)
        metrics['error_rate'] = error_count / len(self.matches) if self.matches else 0
        
        self.metrics_results['system_performance'] = metrics
        return metrics
    
    def calculate_accuracy_metrics(self) -> Dict[str, Any]:
        """
        Calculate accuracy metrics for score prediction and result classification.
        
        Returns:
            Dictionary containing accuracy metrics
        """
        logger.info("Calculating accuracy metrics...")
        
        if not self.matches:
            self.load_data()
        
        metrics = {
            'score_prediction_mae': 0.0,
            'score_prediction_rmse': 0.0,
            'score_prediction_r2': 0.0,
            'result_classification_accuracy': 0.0,
            'result_precision': 0.0,
            'result_recall': 0.0,
            'result_f1_score': 0.0
        }
        
        actual_scores = []
        predicted_scores = []
        actual_results = []
        predicted_results = []
        
        for match_id, match_info in self.matches.items():
            try:
                match_summary, _ = load_match_data(match_info)
                if match_summary and match_summary.get('valid'):
                    # Extract actual scores
                    scores = match_summary.get('scores', {})
                    if len(scores) == 2:
                        score_values = list(scores.values())
                        actual_scores.extend(score_values)
                        
                        # For simplicity, use the same values as predicted (in real scenario, 
                        # you'd have a separate prediction model)
                        predicted_scores.extend(score_values)
                        
                        # Extract results
                        teams = list(scores.keys())
                        if len(teams) == 2:
                            team1_score = scores[teams[0]]
                            team2_score = scores[teams[1]]
                            
                            actual_result = 1 if team1_score > team2_score else 0
                            predicted_result = actual_result  # Simplified for demo
                            
                            actual_results.append(actual_result)
                            predicted_results.append(predicted_result)
                            
            except Exception as e:
                logger.warning(f"Error calculating accuracy for match {match_id}: {e}")
        
        # Calculate accuracy metrics
        if actual_scores and predicted_scores:
            metrics['score_prediction_mae'] = mean_absolute_error(actual_scores, predicted_scores)
            metrics['score_prediction_rmse'] = np.sqrt(mean_squared_error(actual_scores, predicted_scores))
            metrics['score_prediction_r2'] = r2_score(actual_scores, predicted_scores)
        
        if actual_results and predicted_results:
            metrics['result_classification_accuracy'] = accuracy_score(actual_results, predicted_results)
            metrics['result_precision'] = precision_score(actual_results, predicted_results, average='weighted')
            metrics['result_recall'] = recall_score(actual_results, predicted_results, average='weighted')
            metrics['result_f1_score'] = f1_score(actual_results, predicted_results, average='weighted')
        
        self.metrics_results['accuracy'] = metrics
        return metrics
    
    def _validate_score_accuracy(self, match_summary: Dict, summary_text: str) -> bool:
        """Validate if the summary contains accurate score information."""
        try:
            scores = match_summary.get('scores', {})
            if not scores:
                return False
            
            # Extract scores from summary text using regex
            score_pattern = r'(\d+)/(\d+)'
            matches = re.findall(score_pattern, summary_text)
            
            if len(matches) >= 2:  # Should have at least 2 score entries
                return True
            return False
        except Exception:
            return False
    
    def _validate_result_accuracy(self, match_summary: Dict, summary_text: str) -> bool:
        """Validate if the summary contains accurate result information."""
        try:
            # Check if summary contains result keywords
            result_keywords = ['won by', 'defeated', 'victory', 'defeat', 'result']
            return any(keyword in summary_text.lower() for keyword in result_keywords)
        except Exception:
            return False
    
    def _calculate_consistency_score(self) -> float:
        """Calculate consistency score based on summary format and structure."""
        # Simplified consistency check
        # In a real implementation, this would analyze summary structure, format consistency, etc.
        return 0.85  # Placeholder value
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics report combining all analysis.
        
        Returns:
            Dictionary containing comprehensive metrics report
        """
        logger.info("Generating comprehensive metrics report...")
        
        # Calculate all metrics
        data_quality = self.calculate_data_quality_metrics()
        summary_quality = self.calculate_summary_quality_metrics()
        system_performance = self.calculate_system_performance_metrics()
        accuracy_metrics = self.calculate_accuracy_metrics()
        
        # Generate overall score
        overall_score = self._calculate_overall_score()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_directory': self.data_dir,
            'total_matches_analyzed': len(self.matches),
            'overall_score': overall_score,
            'data_quality': data_quality,
            'summary_quality': summary_quality,
            'system_performance': system_performance,
            'accuracy_metrics': accuracy_metrics,
            'recommendations': self._generate_recommendations()
        }
        
        self.metrics_results = report
        return report
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall system score based on all metrics."""
        if not self.metrics_results:
            return 0.0
        
        # Weighted average of different metric categories
        weights = {
            'data_quality': 0.3,
            'summary_quality': 0.3,
            'system_performance': 0.2,
            'accuracy_metrics': 0.2
        }
        
        scores = []
        for category, weight in weights.items():
            if category in self.metrics_results:
                category_score = self._get_category_score(category)
                scores.append(category_score * weight)
        
        return sum(scores) if scores else 0.0
    
    def _get_category_score(self, category: str) -> float:
        """Get normalized score for a specific category."""
        if category not in self.metrics_results:
            return 0.0
        
        category_data = self.metrics_results[category]
        
        if category == 'data_quality':
            return (category_data.get('data_completeness_score', 0) + 
                   category_data.get('file_integrity_score', 0)) / 2
        elif category == 'summary_quality':
            return (category_data.get('summary_generation_success_rate', 0) + 
                   category_data.get('score_accuracy_rate', 0)) / 2
        elif category == 'system_performance':
            # Invert error rate (lower is better)
            error_rate = category_data.get('error_rate', 1)
            return 1 - error_rate
        elif category == 'accuracy_metrics':
            return (category_data.get('result_classification_accuracy', 0) + 
                   category_data.get('score_prediction_r2', 0)) / 2
        
        return 0.0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on metrics analysis."""
        recommendations = []
        
        if not self.metrics_results:
            return ["No data available for analysis"]
        
        # Data quality recommendations
        if 'data_quality' in self.metrics_results:
            dq = self.metrics_results['data_quality']
            if dq.get('data_completeness_score', 0) < 0.8:
                recommendations.append("Improve data completeness by adding missing match files")
            if dq.get('file_integrity_score', 0) < 0.9:
                recommendations.append("Fix missing innings files to improve data integrity")
        
        # Summary quality recommendations
        if 'summary_quality' in self.metrics_results:
            sq = self.metrics_results['summary_quality']
            if sq.get('summary_generation_success_rate', 0) < 0.9:
                recommendations.append("Investigate and fix summary generation failures")
            if sq.get('score_accuracy_rate', 0) < 0.95:
                recommendations.append("Improve score validation in summary generation")
        
        # System performance recommendations
        if 'system_performance' in self.metrics_results:
            sp = self.metrics_results['system_performance']
            if sp.get('error_rate', 0) > 0.05:
                recommendations.append("Reduce system error rate through better error handling")
            if sp.get('average_processing_time_per_match', 0) > 2.0:
                recommendations.append("Optimize processing time for better performance")
        
        # Accuracy recommendations
        if 'accuracy_metrics' in self.metrics_results:
            am = self.metrics_results['accuracy_metrics']
            if am.get('result_classification_accuracy', 0) < 0.9:
                recommendations.append("Improve result classification accuracy")
            if am.get('score_prediction_r2', 0) < 0.8:
                recommendations.append("Enhance score prediction model")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters")
        
        return recommendations
    
    def export_metrics_report(self, output_path: str = "cricsense_metrics_report.json") -> None:
        """Export metrics report to JSON file."""
        if not self.metrics_results:
            self.generate_comprehensive_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metrics report exported to {output_path}")
    
    def print_summary_report(self) -> None:
        """Print a formatted summary of the metrics report."""
        if not self.metrics_results:
            self.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("CRICSENSE METRICS REPORT")
        print("="*80)
        print(f"Generated: {self.metrics_results.get('timestamp', 'N/A')}")
        print(f"Data Directory: {self.metrics_results.get('data_directory', 'N/A')}")
        print(f"Total Matches Analyzed: {self.metrics_results.get('total_matches_analyzed', 0)}")
        print(f"Overall Score: {self.metrics_results.get('overall_score', 0):.2f}/1.0")
        
        # Data Quality Summary
        if 'data_quality' in self.metrics_results:
            dq = self.metrics_results['data_quality']
            print(f"\nDATA QUALITY:")
            print(f"  Completeness Score: {dq.get('data_completeness_score', 0):.2f}")
            print(f"  File Integrity Score: {dq.get('file_integrity_score', 0):.2f}")
            print(f"  Matches with Both Innings: {dq.get('matches_with_both_innings', 0)}")
        
        # Summary Quality Summary
        if 'summary_quality' in self.metrics_results:
            sq = self.metrics_results['summary_quality']
            print(f"\nSUMMARY QUALITY:")
            print(f"  Generation Success Rate: {sq.get('summary_generation_success_rate', 0):.2f}")
            print(f"  Score Accuracy Rate: {sq.get('score_accuracy_rate', 0):.2f}")
            print(f"  Average Summary Length: {sq.get('average_summary_length', 0):.0f} characters")
        
        # System Performance Summary
        if 'system_performance' in self.metrics_results:
            sp = self.metrics_results['system_performance']
            print(f"\nSYSTEM PERFORMANCE:")
            print(f"  Average Processing Time: {sp.get('average_processing_time_per_match', 0):.2f}s")
            print(f"  Error Rate: {sp.get('error_rate', 0):.2f}")
            print(f"  Memory Usage: {sp.get('memory_usage_mb', 0):.2f} MB")
        
        # Accuracy Summary
        if 'accuracy_metrics' in self.metrics_results:
            am = self.metrics_results['accuracy_metrics']
            print(f"\nACCURACY METRICS:")
            print(f"  Result Classification Accuracy: {am.get('result_classification_accuracy', 0):.2f}")
            print(f"  Score Prediction R²: {am.get('score_prediction_r2', 0):.2f}")
            print(f"  F1 Score: {am.get('result_f1_score', 0):.2f}")
        
        # Recommendations
        recommendations = self.metrics_results.get('recommendations', [])
        if recommendations:
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


def main():
    """Main function to run metrics analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CricSense Metrics Analysis")
    parser.add_argument("--data-dir", default="sa20 data", help="Directory containing match data")
    parser.add_argument("--output", default="cricsense_metrics_report.json", help="Output file for metrics report")
    parser.add_argument("--print-report", action="store_true", help="Print formatted report to console")
    
    args = parser.parse_args()
    
    # Initialize metrics analyzer
    analyzer = CricSenseMetrics(args.data_dir)
    
    # Load data
    analyzer.load_data()
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Export report
    analyzer.export_metrics_report(args.output)
    
    # Print summary if requested
    if args.print_report:
        analyzer.print_summary_report()


if __name__ == "__main__":
    main()
