function convertResultsToTrx(infile,frontout,sideout)

%%
Q = load(infile);


%%

ts = now;
Jf = struct;
nd = size(Q.pfrontbest,1);
npts = size(Q.pfrontbest,2);
nfrms = size(Q.pfrontbest,3);
Jf.pTrk = permute(Q.pfrontbest,[2 1 3 4]);
Jf.pTrkTS = repmat(ts,[npts,nfrms,1]);
Jf.pTrkTag = cell([npts,nfrms,1]);

Js = struct;
Js.pTrk = permute(Q.psidebest,[2 1 3 4]);
Js.pTrkTS = repmat(ts,[npts,nfrms,1]);
Js.pTrkTag = cell([npts,nfrms,1]);

%%

save(frontout,'-struct','Jf');
save(sideout,'-struct','Js');

