from rouge_score import rouge_scorer


def calculate_rouge_scores(reference, hypothesis):
    rouge_scores = {}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    rouge_scores['rouge1'] = scores['rouge1'].fmeasure
    rouge_scores['rouge2'] = scores['rouge2'].fmeasure
    rouge_scores['rougeL'] = scores['rougeL'].fmeasure
    return rouge_scores


def evaluate_summary_performance(reference_texts, generated_summaries):
    rouge_scores = []
    for reference, summary in zip(reference_texts, generated_summaries):
        rouge_score = calculate_rouge_scores(reference, summary)
        rouge_scores.append(rouge_score)
    avg_rouge_scores = {
        'rouge1': sum(score['rouge1'] for score in rouge_scores) / len(rouge_scores),
        'rouge2': sum(score['rouge2'] for score in rouge_scores) / len(rouge_scores),
        'rougeL': sum(score['rougeL'] for score in rouge_scores) / len(rouge_scores)
    }
    return avg_rouge_scores


def calculate_accuracy(sentiments, predicted_sentiments):
    "Calculate the accuracy of the sentiment classification."
    correct = 0
    for sentiment, predicted_sentiment in zip(sentiments, predicted_sentiments):
        if sentiment == predicted_sentiment:
            correct += 1
    accuracy = correct / len(sentiments)
    return accuracy