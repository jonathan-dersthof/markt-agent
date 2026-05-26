  # Hybrid-DRL/SL-Modell zur Aktienanalyse

Im Rahmen dieses Projekts habe ich mir vorgenommen direkt im Anschluss an meine [Facharbeit] meine Kenntnisse auszubauen und weiterzulernen. Meine zwei Hauptmotivationen für die Auswahl dieses spezifischen Projekts sind, dass ich mich praktisch mit Supervised Learning auseinandersetzen wollte, da ich mich bisher nur mit Reinforcement Learning beschäftigte und dass ich mit realen Daten und nicht nur mit "künstlichen Daten" wie im vorherigen Projekt arbeiten wollte.
***
  - Mein Ziel
  
Ich möchte als kleines Experiment einen DRL Agenten trainieren, der simuliert mit Aktien zu handeln. Dieser Agent soll als einen Input Wert eine vorhersage eines Supervised Learning Models (LSTM) erhalten, ob der Kurs steigt, fällt oder seitwärts vorläuft. 
***
  - Aktueller Stand (26.05.2026)

Aktuell arbeite ich als erstes an dem LSTM Modell. Ich habe eine grobe Struktur für das Projekt geschaffen und es werden Finanzdaten mit yfinance in Python importiert. In den ersten Experimenten habe ich gemerkt, dass das Modell mit rein technischen Daten nur schwer bessere performances zu erreichen scheint, als pures raten. Die confidence in den Vorhersagen schwankt zwischen 20-40%. In der nächsten Zeit möchte ich einige Konstellationen der Hyperparameter sowie Features durchrotieren, um zu schauen, ob es doch möglich ist mit den rein technischen Daten ein Modell zu erhalten, das besser als Raten funktioniert. Sollte das nicht klappen habe ich mir folgende Alternativen überlegt: 

A) Statt das Modell mit mehreren Tickern zu trainieren veruche ich einen Trainingsablauf zu schaffen, in dem für jeden Ticker ein spezialisiertes Modell trainiert wird. 

B) Neben den rein technischen Daten versuche ich einen Weg zu finden zuverlässig an historische/aktuelle Fundamentaldaten zu kommen. Da die technischen Daten bei yfinance nur täglich aktualisiert werden und die Daten für einen Tag enthalten muss der Agent in der Simulation langfristiger Investieren wofür Fundamentaldaten wichtiger sind. 

C) Ich könnte auch versuchen technische Daten zu finden, welche enger getaktet sind, um den Agenten zu einem "Daytrader" zu machen, wofür technische Daten wohl eher ausreichen könnten auch ohne Fundamentaldaten. 

So oder so muss ich weiterhin recherchieren wie ich Rauschen aus den Daten filtern kann. 
