from rouge_score import rouge_scorer
import sys

# --- Import functions from your main script ---
# Make sure your main script is named 'cricsense_match_summary.py'
# and is in the same folder as this evaluate_baseline.py file.
try:
    from cricsense_match_summary import get_match_files, load_match_data, generate_match_summary, build_structured_summary
except ImportError:
    print("Error: Could not import functions from 'cricsense_match_summary.py'.")
    print("Please ensure both scripts are in the same directory.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

# ==========================================================
#  CONFIGURATION - YOU MUST EDIT THIS SECTION!
# ==========================================================
DATA_DIR = "sa20 data"  # Make sure this points to your data folder

# 1. Choose a real Match ID from your data
MATCH_ID_TO_TEST = "1343941" # <--- CHANGE THIS to a valid Match ID

# 2. Run your main script for the chosen ID and PASTE ITS SUMMARY HERE:
#    Example command: python cricsense_match_summary.py --match-id 1343941
GENERATED_SUMMARY = """
====================================================================================================
                     BETWAY SA20 LEAGUE 2022/23 : Paarl Royals vs MI Cape Town                     
====================================================================================================
                             DATE: 10-01-2023                             
====================================================================================================

MATCH SUMMARY:
In an exciting clash, Paarl Royals scored 142/7 in their innings. In response, MI Cape Town chased the target, finishing with a score of 143/2. Ultimately, MI Cape Town won by 8 wickets.

====================================================================================================

SCORECARD:
====================================================================================================
1st Innings: Paarl Royals
  142/7 in 20.0 overs (Extras: 7)

2nd Innings: MI Cape Town (Target: 143)
  143/2 in 15.3 overs (Extras: 6)

====================================================================================================
                         RESULT: MI Cape Town won by 8 wickets (with 27 balls remaining)                         
====================================================================================================
""" # <--- PASTE THE OUTPUT OF YOUR SCRIPT BETWEEN THE TRIPLE QUOTES

# 3. Find the official HUMAN-WRITTEN summary for the SAME match online and PASTE IT HERE:
REFERENCE_SUMMARY = """
MI Cape Town cruise to victory in SA20 opener. Dewald Brevis starred with an unbeaten 70 as MI Cape Town chased down Paarl Royals' 142 for 7 with ease, winning by eight wickets with 27 balls to spare in the inaugural SA20 match at Newlands. Jofra Archer also impressed on his return, taking 3 for 27 for MI Cape Town. Jos Buttler top-scored for the Royals with 51.
""" # <--- PASTE THE HUMAN SUMMARY (e.g., from ESPNcricinfo) HERE

# ==========================================================

print(f"Evaluating Baseline Summary for Match ID: {MATCH_ID_TO_TEST}...")

# --- Optional: You can uncomment these lines if you want the script ---
# --- to dynamically generate the summary instead of pasting it above ---
# print("Loading match data to generate summary dynamically...")
# all_matches = get_match_files(DATA_DIR)
# if MATCH_ID_TO_TEST not in all_matches:
#     print(f"Error: Match ID {MATCH_ID_TO_TEST} not found in data directory '{DATA_DIR}'.")
#     sys.exit(1)
# match_info = all_matches[MATCH_ID_TO_TEST]
# match_summary_stats, _ = load_match_data(match_info)
# if not match_summary_stats or not match_summary_stats['valid']:
#     print("Error: Could not load valid match data to generate summary.")
#     sys.exit(1)
# structured_data = build_structured_summary(match_info, match_summary_stats)
# GENERATED_SUMMARY = structured_data.get('summary_text', "Error: Could not dynamically generate summary.")
# print("Dynamically generated summary obtained.")
# --- End of optional dynamic generation ---


# --- Calculate ROUGE Scores ---
print("\nCalculating ROUGE scores...")
try:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(REFERENCE_SUMMARY, GENERATED_SUMMARY)
except Exception as e:
    print(f"An error occurred while calculating ROUGE scores: {e}")
    sys.exit(1)

# --- Print the Results ---
print("\n" + "="*40)
print("  BASELINE MODEL EVALUATION METRICS")
print("="*40)

print("\n--- REFERENCE (HUMAN) SUMMARY ---")
print(REFERENCE_SUMMARY.strip())

print("\n--- GENERATED (BASELINE) SUMMARY ---")
# Let's print only the narrative part for cleaner comparison
narrative_part = GENERATED_SUMMARY.split("MATCH SUMMARY:")[1].split("SCORECARD:")[0].strip()
print(narrative_part)


print("\n" + "="*40)
print("         ROUGE SCORE RESULTS")
print("="*40)

print("\nROUGE-1 (Word Overlap):")
print(f"  Precision: {scores['rouge1'].precision:.4f}")
print(f"  Recall:    {scores['rouge1'].recall:.4f}")
print(f"  F1-Score:  {scores['rouge1'].fmeasure:.4f}\n")

print("ROUGE-2 (Word-Pair Overlap):")
print(f"  Precision: {scores['rouge2'].precision:.4f}")
print(f"  Recall:    {scores['rouge2'].recall:.4f}")
print(f"  F1-Score:  {scores['rouge2'].fmeasure:.4f}\n")

print("ROUGE-L (Longest Common Subsequence):")
print(f"  Precision: {scores['rougeL'].precision:.4f}")
print(f"  Recall:    {scores['rougeL'].recall:.4f}")
print(f"  F1-Score:  {scores['rougeL'].fmeasure:.4f}\n")

print("="*40)