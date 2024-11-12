%% Fase di INPUT (1): acquisisce l'immagine iperspettrale
hcube = hypercube('taglio_Mola_EL1.img', 'taglio_Mola_EL1.hdr');

%% Fase di INPUT (2): acquisisce l'immagine in RGB
imgRGB = colorize(hcube, 'Method', 'rgb', 'ContrastStretching', true);

%% Fase di acquisizione degli indici NDVI
indexNDVI = ndvi(hcube);

%% Fase di acquisizione delle firme spettrali
numEndmembers = countEndmembersHFC(hcube.DataCube(1700:2000, 2200:2600, :)); % individua il numero di endmembers di una determinata scena dell'immagine
endmembers = nfindr(hcube.DataCube(1700:2000, 2200:2600, :),numEndmembers); % individua gli endmembers utilizzando l'algoritmo nfindr

%% Fase di stampa delle firme spettrali per ciascun endmember
figure
plot(endmembers)
legend('Location','Bestoutside')


%% Fase di CLASSIFICAZIONE 

%{

La matrice scores ha dimensione 4293x4890x8 (4293x4890 = dimensione dell’intera immagine di input; 8 = numero di endmembers).
Questa sintassi permette di individuare gli 8 endmembers per tutta l'immagine.
La matrice, contiene la similarità spettrale presente tra lo spettro dei pixel dell’hcube e lo spettro degli endmembers. 

%}

scores = zeros(size(hcube.DataCube, 1), size(hcube.DataCube, 2), numEndmembers); 
for i = 1:numEndmembers
    scores(:, :, i) = sam(hcube, endmembers(:,i)); % applica l'algoritmo SAM 
end

%% Fase di PREDIZIONE 

%{

Si usa la funzione min(scores,[],3);  per associare un indice numerico ad ogni pixel dell'immagine. Ad esempio, ad un pixel di coordinate (A,B), verrà associato il valore 7 se è stato identificato come pixel di ulivo.  

indexOK varra' 1 quando e' rispettata la condizione (indexOliveTree == 7) & (indexNDVI>=0.60) & (indexNDVI<=0.80 ; 0 altrimenti. Questa sintassi serve per indicare se il pixel e' effettivamente un pixel di ulivo (ha l'etichetta 7) e se ha indice NDVI congruo a quello di un ulivo (0.6<=indice ndvi ulivo<=0.8).

Si applica la funzione reshape affinché la logica 0/1 venga applicata sull'immagine iperspettrale di input.

La funzione imresize modifica le dimensioni di un’immagine per adattarla al formato più idoneo. 

La funzione imoverlay permette di sovrapporre due immagini: viene utilizzata per sommare l’immagine a delle aree bianche in modo tale che i pixel di ulivo vengano colorati appunto di bianco (il bianco è identificato dal parametro impostato ad [1, 1, 1] all’interno della funzione)

%}

[~,indexOliveTree] = min(scores,[],3); 
indexOK = (indexOliveTree == 7) & (indexNDVI>=0.60) & (indexNDVI<=0.80);
indexOKreshaped = reshape(indexOK, size(hcube.DataCube, 1), size(hcube.DataCube, 2));
indexOKresized = imresize(indexOKreshaped, size(imgRGB, 1:2));
imgIndexOKonRGB = imoverlay(imgRGB, indexOKresized, [1 1 1]);

%% Fase di OUTPUT: stampa dell’immagine di input con i pixel identificati come di ulivo in bianco
figure('Position',[0 0 1100 500])
subplot('Position',[0 0.15 0.4 0.8])
imagesc(imgRGB)
axis off
title('RGB Image of Hyperspectral Data')
subplot('Position',[0.45 0.15 0.4 0.8])
imagesc(imgIndexOKonRGB)
axis off
title('Indices of Matching Endmembers')



