"""Build the labels file that audit.py reads from the three shipped label files of 14 and 15 Nov (and the 15 Nov night run) (labeller A, high effort).
Later file wins if an (answer file, question) pair appears in both. UNLABELLED becomes UNPARSED. No answer text is involved.
Usage: python build_labels_tashon.py OUT.csv   (run from the repository root)"""
import csv, sys
rows = {}
for p in ("nov14_v2/labels/labels_high_effort.csv", "nov15_overnight/labels/labels_high_effort_nov15.csv",
          "nov15_overnight/night2/labels/labels_high_effort_night2.csv"):
    for r in csv.DictReader(open(p)): rows[(r["answer_file"], r["question_id"])] = r["label_high_effort"]
with open(sys.argv[1], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["arm", "question_id", "label", "hit_max", "chars"])
    for (a, q), l in rows.items(): w.writerow([a, q, "UNPARSED" if l == "UNLABELLED" else l, 0, 0])
print(len(rows), "labels,", len({a for a, _ in rows}), "answer files")
