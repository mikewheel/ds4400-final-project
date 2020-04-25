"""
Code for generating ROC curves from a trained classifier.
Used https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html as example code.

Written by Michael Wheeler and Jay Sherman.
"""

from bokeh.plotting import figure, output_file, save
from sklearn.metrics import roc_curve, auc

from config import make_logger

logger = make_logger(__name__)


def plot_roc_auc(model, title, x_test, y_test, output_dir) -> None:
    output_file(output_dir / f'roc_{title}.html')
    
    y_predictions = model.predict(x_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(1):
        fpr[i], tpr[i], thresholds = roc_curve(y_test, y_predictions)
        roc_auc[i] = auc(fpr[i], tpr[i])

    p = figure(title=f'ROC for {title}', plot_width=800, plot_height=800, x_range=(0, 1.05), y_range=(0, 1.05))
    p.xlabel('False Positive Rate')
    p.ylabel('True Positive Rate')
    
    p.line([0, 1], [0, 1], line_width=1, line_color="blue")
    p.line(fpr[0], tpr[0], line_width=2, line_color="orange", legend_label=f'white wine: area = {round(roc_auc[0], 2)}')
    save(p)
