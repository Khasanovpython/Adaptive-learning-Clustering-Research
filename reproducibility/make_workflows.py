"""Draw the study design and comparison paths in Figures 1 and 3."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / 'results'

def draw(filename, boxes, arrows):
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    ax.set(xlim=(0, 11), ylim=(0, 5))
    ax.axis('off')
    for x,y,w,h,caption,color in boxes:
        box = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.10',
                            edgecolor='#34465c', facecolor=color, linewidth=1.05)
        ax.add_patch(box)
        ax.text(x+w/2,y+h/2,caption,ha='center',va='center',fontsize=11,color='#172839')
    for start,end in arrows:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,
                                     color='#425c76',shrinkA=0,shrinkB=0,lw=1.05))
    fig.savefig(OUT/filename,dpi=450,bbox_inches='tight',pad_inches=.06,facecolor='white')
    plt.close(fig)

def main():
    blue, green, ivory = '#eaf1f9','#edf6ef','#faf3e5'
    draw('Figure_1_Research_Design.png',[
        (.3,3.6,3.0,1.0,'Documented generator\nSeed 42 · 4,600 profiles',blue),
        (4.0,3.6,3.0,1.0,'Clean primary benchmark\nTen features · four scenarios',blue),
        (7.7,3.6,3.0,1.0,'Stress conditions\nNoise, overlap, imbalance,\ndependence, count, outsiders',ivory),
        (4.0,1.85,6.7,1.0,'Matched comparisons\nClassical · soft-only · hybrid\nRule-only · crisp rules · fuzzy c-means',green),
        (.3,.25,3.0,1.0,'Legacy workbook\nSupplementary comparison',ivory),
        (4.0,.25,6.7,1.0,'Evaluation\nScenario recovery · coverage · internal indices',blue),
    ], [((3.35,4.1),(3.95,4.1)),((7.05,4.1),(7.65,4.1)),((5.5,3.55),(5.5,2.9)),
        ((9.2,3.55),(9.2,2.9)),((7.35,1.8),(7.35,1.3)),((3.35,.75),(3.95,.75))])
    draw('Figure_3_Comparison_Framework.png',[
        (.3,3.6,3.0,1.0,'Original-scale features',blue),
        (4.0,3.6,3.0,1.0,'Standardized features',blue),
        (7.7,3.6,3.0,1.0,'Standardized features',blue),
        (.3,1.9,3.0,1.0,'Four scenario rules\nTriangular / trapezoidal terms',ivory),
        (4.0,1.9,3.0,1.0,'Classical partitions\nSoft cluster memberships',ivory),
        (7.7,1.9,3.0,1.0,'Fuzzy c-means\nData-fitted prototypes',ivory),
        (.3,.25,3.0,1.0,'Direct fuzzy and crisp\nrule-only assignments',green),
        (4.0,.25,3.0,1.0,'Hybrid assignment\n0.8 cluster + 0.2 rule',green),
        (7.7,.25,3.0,1.0,'Dominant membership\nFuzzy c-means assignment',green),
    ], [((1.8,3.55),(1.8,2.95)),((5.5,3.55),(5.5,2.95)),((9.2,3.55),(9.2,2.95)),
        ((1.8,1.85),(1.8,1.3)),((5.5,1.85),(5.5,1.3)),((9.2,1.85),(9.2,1.3)),((3.35,2.04),(3.95,.94))])

if __name__ == '__main__':
    main()
