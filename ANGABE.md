# GKXXXX Data Science "NER Classification" (POSA)

## Data Science "NER Classification" - Taskdescription

### Einführung

Es ist empfehlenswert, dass unten verlinkte [Tutorial](https://huggingface.co/learn/nlp-course/chapter7/2) als Vorlage und Hilfestellung zu verwenden.

### Ziele

Das Ziel ist es, das Sprachmodell `bert-base-cased` für Named Entity Recognition (NER) zu feintunen. Im Endeffekt soll das Modell zwischen Personen, Organisationen, Orten und MISC klassifizieren können. Dazu soll der [conllpp-Datensatz](https://huggingface.co/datasets/conllpp) verwendet werden.

### Voraussetzungen

- Kenntnis von Markdown, LaTeX und Grundkenntnisse in R und Python
- funktionstüchtige Installation von R, RStudio, Python auf eurem Rechner, virtueller Maschine etc.
- Kenntnis von grundlegenden Methoden der Wahrscheinlichkeitstheorie.
- Die Verwendung von Jupyter Notebooks ist empfehlenswert, aber nicht erforderlich.

Außerdem werden folgende Python-Libraries benötigt:

```shell
pip install torch --index-url https://download.pytorch.org/whl/cpu  # Wenn eine CPU verwendet wird und keine NVIDIA-GPU vorhanden ist
pip install torch  # Wenn eine NVIDIA-GPU vorhanden ist
pip install datasets transformers seqeval numpy
```

### Aufgabe

Das Sprachmodell Bert soll mittels dem [conllpp-Datensatz](https://huggingface.co/datasets/conllpp) auf Named Entity Recognition gefeintuned werden. Das fertige Modell soll dabei Personen, Organisationen, Orte und MISC klassifizieren können. Gehe dabei wie folgt vor:

1. Lade den [conllpp-Datensatz](https://huggingface.co/datasets/conllpp) aus dem HuggingFace-Repository und den Tokenizer für das Sprachmodell `bert-base-cased` in deinem Notebook herunter.
2. Lade das Sprachmodell `bert-base-cased` herunter. Achtung: Die Anzahl der NER-TAGS und die Zuordnung von den NER-TAGS zu den Indezes und umgekehrt müssen beim Laden des Modells mitgegeben werden.
3. Erstelle eine Funktion `tokenize_and_align_labels` wie im Tutorial. Diese soll im Endeffekt (In Kombination mit der Funtion `map`) den gesamten Datensatz tokenizen. Zusätzlich soll die Methode Labels (=NER-TAGS) und den entsprechenden Tokens zuweisen. Da wir Tokens klassifizieren, und ein Wort oft in mehrere Tokens aufgeteilt wird, müssen wir auch zwischen dem ersten Token eines Wort und den anderen unterscheiden. Das ist wegen der Fehlerberechnung notwendig, da wir nicht mehrere Tokens desselben Worts berücksichtigen wollen. Deshalb muss die Funktion auch diese Vorgehensweise implementieren.
4. Erstelle einen Data Collator, der dafür sorgt, dass jeder Input im selben Batch dieselbe Länge hat.
5. Erstelle die Fehlerfunktion.
6. Trainiere das Modell mittels der `Trainer`-Klasse. Je nach dem, ob man eine NVIDIA-Grafikkarte besitzt oder nicht, ist es empfehlenswert, mit den *Hyperparametern* `learning_rate`, `batch_size` und `num_train_epochs` herumzuspielen.
7. Teste dein Modell mit einer *pipeline* aus der *transformers*-Library.

### Abgabe

Das Protokoll ist als PDF-Dokument und Markdown-File mit ausführbarem Code abzugeben.

### Bewertung

Gruppengrösse: 1 Person

### Anforderungen überwiegend erfüllt

- Aktuelle Markdown- oder LaTeX-Protokollvorlage aus Github bzw. Moodle verwendet
- Grundlegende Beschreibung aller wichtigen Begriffe und Methoden
- Dokumentation aller Arbeitsschritte

### Anforderungen vollständig erfüllt

- Verwendung über das Mindestmaß hinausgehender Methoden und Algorithmen
- Verbale Beschreibung und Erklärung aller angeführter Begriffe und deren Anwendung in konkreten Beispielen in vollständigen deutschen Sätzen
- Funktionierendes NER-Modell

### Quellen

https://huggingface.co/learn/nlp-course/chapter7/2
