# Zeilen, die mit ‚#‘ beginnen, sind Kommentare und werden ignoriert. Kommentare 
# dürfen auch am Zeilenende auftreten. Leere Zeilen werden ebenfalls ignoriert

8,5=0  # die Größenangabe kann auch ‚Breite,Höhe‘ sein. Hinter ‚=‘ kann angegeben werden, welchen 
# Ursprung die Koordinaten haben (standardmäßig 1, wie in den Beispieldaten; da numpy-Arrays aber 
# mit Index 0 beginnen, ist die Angabe ‚=0‘ für konsistente Werte sinnvoll)

0,0,4  # Roboterbatterie

*  # Die Anzahl der Batterien darf auch unbekannt sein. 
2,3,1:BATTERIE LABEL  # Batterien dürfen Beschriftungen nach ‚:‘ haben
# […]
# Ein Teleportationsfeldpaar wird folgendermaßen definiert (x1,y1,x2,y2):
T1,1,2,2
-  # wenn es weitere Zeilen gibt, muss das Ende der Batterien durch ‚-‘ gekennzeichnet werden

LABELS  # spezifiziert, dass nun weitere Beschriftungen für Felder (auch ohne Batterien) folgen
4,1:ICH BIN EINE BESCHRIFTUNG  # Format: x,y:Text
-  # s.o.
# am Ende darf sich eine mögliche Lösung für das Spielfeld befinden (vorgesehener Weg wird hier von 
# aufgabe1b.py eingetragen, sodass er in der SVG-Datei mit Funktion save_svg eingezeichnet werden 
# kann)
#!solution=x1,y1;x2;y2;...
