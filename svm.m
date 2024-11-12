%% Fase di INPUT (1): acquisisce l'immagine iperspettrale
hcube = hypercube('taglio_Mola_EL1.img', 'taglio_Mola_EL1.hdr');

%% Fase di INPUT (2): acquisisce l'immagine in RGB
imgRGB = colorize(hcube, 'Method', 'rgb', 'ContrastStretching', true);

%% Fase di acquisizione degli indici NDVI
indexNDVI = ndvi(hcube);

%% Fase di acquisizione dei 25 pixel (visualizza l'immagine a schermo) 

%{

la funzione zoom on permette di zoommare l’immagine presentata a video; 
premendo un pulsante qualsiasi, l’utente può selezionare, mediante il cursore del mouse, i 25 pixel da acquisire;
al termine dell’acquisizione, lo zoom viene ripristinato;

%}

numPX=25; % numero di pixel da acquisire in input = numero di campioni che dovranno essere utilizzati per il training del modello
figure % la funzione figure visualizza a schermo l’immagine di input...
imagesc(imgRGB); %...in RGB
axis image off
title('RGB Image of Data Cube') % stampa a video il titolo dell’immagine
zoom on; % applica lo zoom
pause(); % lo zoom resta applicato e permette all'utente di selezionare i pixel; premendo un pulsante qualsiasi della tasitera...
zoom off; %... lo zoom viene rimosso

%% Fase di salvataggio delle coordinate dei punti selezionati
[x,y] = ginput(numPX); %  la funzione ginput permette di memorizzare, all’interno di un array bidimensionale [x;y], le coordinate di ogni punto selezionato
x = round(x); % arrotonda il valore numerico delle coordinate x, dell’array bidimensione [x;y], all’intero più vicino
y = round(y); % arrotonda il valore numerico delle coordinate y, dell’array bidimensione [x;y], all’intero più vicino

%% Fase di acquisizione delle firme spettrali
spectralSignaturesPX = zeros(numPX, size(hcube.DataCube, 3)); % struttura dati, inizializzata a zero, di 25x47 (predisposta per contenere le firme spettrali delle 47 bande dei 25 px acquisiti in input)
classesNDVI = zeros(numPX, 1);
for i = 1:numPX % ciclo for che itera tante volte quanto è il numero di punti selezionati:
    xPX = x(i); % memorizza in una variabile xi l’i-esima coordinata x dell’array bidimensione [x;y];
    yPX = y(i); % arrotonda il valore numerico delle coordinate y, dell’array bidimensione [x;y], all’intero più vicino;

    % Ottiene la firma spettrale del pixel selezionato
    spectralSignaturePX = squeeze(hcube.DataCube(yPX, xPX, :));
	% in spectralSignaturePX verrà memorizzato esclusivamente l’elemento di dimensione 47 (terza componente del DataCube), composto dalle 47 firme spettrali dello specifico pixel
    
    % Salva la firma spettrale nella matrice delle firme spettrali
    spectralSignaturesPX(i, :) = spectralSignaturePX; %permette il salvataggio delle firme spettrali dell’i-esimo pixel nel vettore delle firme spettrali di tutti i 25 pixel
	
    classesNDVI(i) = indexNDVI(yPX,xPX); % si utilizza l’indice NDVI come discriminante per la suddivisione in classi per l’addestramento del classificatore. Mediante questa sintassi, l’indice NDVI dell’i-esimo pixel viene memorizzato nel vettore degli indici NDVI di tutti i 25 pixel
end

%% Fase di CLASSIFICAZIONE 

%{

all’interno del vettore delle classi (classes), si memorizza 1 quando la condizione è rispettata, 0 altrimenti. Si utilizzano 0 e 1 come etichette per identificare due classi, rispettivamente “ulivo”, “non ulivo”.

%}

classes = classesNDVI > 0.8 & classesNDVI < 0.6;

%{

La funzione fitcsvm allena un SVM per una classificazione binaria: 
- il campo 'KernelFunction', valorizzato a ‘linear’ identifica il classificatore come di tipo lineare 
- la fitcsvm acquisisce in input le firme spettrali dei 25 pixel, usate come training set, e le rispettive classi

Tra i 25 pixel presenti, viene assegnata la classe 0 ad ogni pixel con indice NDVI inferiore a 0.8 e maggiore di 0.6 (0.6<=indice ndvi ulivo<=0.8), per identificarlo come pixel di ulivo; vengono usate come ulteriori discriminanti per la previsione, le 47 bande corrispondenti a quel pixel. 

I risultati vengono memorizzati all’interno della struttura dati classifier.

%}

classifier = fitcsvm(spectralSignaturesPX, classes, 'KernelFunction', 'linear', 'Standardize', true); % allena classificatore SVM

%% Fase di PREDIZIONE 

%{

Il classificatore, dopo aver utilizzato come training set gli specifici 25 pixel nel blocco di codice precedente ("allena classificatore SVM"), esegue in quest'altro blocco di codice ("predizione") la sua predizione sull’intera immagine iperspettrale di input: ciò è permesso mediante la funzione predict che acquisisce in input:

1.	le firme spettrali di tutti i pixel dell’immagine (indexOKreshaped);
2.	il modello classifier, che dovrà occuparsi della previsione sull’immagine.

%}

imgReshaped = reshape(hcube.DataCube, [], size(hcube.DataCube, 3));
predictedPX = predict(classifier, imgReshaped);
indexOK = predictedPX == 1;
indexOKreshaped = reshape(indexOK, size(hcube.DataCube, 1), size(hcube.DataCube, 2));
indexOKresized = imresize(indexOKreshaped, size(imgRGB, 1:2));
imgIndexOKonRGB = imoverlay(imgRGB, indexOKresized, [1 1 1]);

%% Fase di OUTPUT: stampa dell’immagine di input con i pixel identificati come di ulivo in bianco

figure;
imshow(imgIndexOKonRGB);
title('Ulivi predetti');

