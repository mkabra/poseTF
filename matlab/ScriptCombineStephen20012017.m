Q = load('../headTracking/FlyHeadStephenCuratedData.mat');

ldir = '/home/mayank/Dropbox/PoseEstimation/Stephen/18012017_trainingData/justHead/';
dd = dir([ldir '*.lbl']);
P = load('/home/mayank/Dropbox/PoseEstimation/Stephen/18012017_trainingData/justHead/fly90.lbl','-mat');


%%
% in Q.pts, view is the second co-ordinate. i.e., Q.pts(:,1,:,:)
% corresponds to side view


local = true;
J = struct;

nexp = numel(Q.vid1files);
movf = cell(nexp,2);
lpos = cell(nexp,1);
lmarked = cell(nexp,1);
lothers = cell(nexp,1);
for ndx = 1:nexp
  if ~local,
    v1f = ['/groups/branson/bransonlab/mayank/' Q.vid1files{ndx}(19:end)];
    v2f = ['/groups/branson/bransonlab/mayank/' Q.vid2files{ndx}(19:end)];
  else
    v1f = Q.vid1files{ndx};
    v2f = Q.vid2files{ndx};
  end
  movf{ndx,1} = v1f;
  movf{ndx,2} = v2f;
  
  [rfn,nframes,fid,hinfo] = get_readframe_fcn(Q.vid1files{ndx});
  curidx = find(Q.expidx==ndx);
  pside = permute(Q.pts(:,1,:,curidx),[3,1,4,2]);  
  pfront = permute(Q.pts(:,2,:,curidx),[3,1,4,2]);
  curpos = nan(10,2,nframes);
  curt = Q.ts(curidx);
  curpos(1:5,:,curt) = pside;
  curpos(6:10,:,curt) = pfront;
  lpos{ndx} = curpos;
  curm = false(10,nframes);
  curm(:,curidx) = true;
  lmarked{ndx} = curm;
  
  if fid>0,fclose(fid); end
end

%%
for ndx = 1:numel(dd)
  P = load(fullfile(ldir,dd(ndx).name),'-mat');
  nexp = numel(P.labeledpos);  
  
  K = cell(nexp,2);
  for ne = 1:nexp
    for vv = 1:2
        kk = strrep(P.movieFilesAll{ne,vv},'\','/');
        if P.movieFilesAll{ne,1}(1) == '$',
          K{ne,vv} = ['/groups/huston/hustonlab/' kk(10:end)];
        else
          K{ne,vv} = ['/groups/huston/hustonlab/' kk(4:end)];
        end
    end    
    l1 = any(~isnan(squeeze(P.labeledpos{ne}(:,1,:))),1);
    l2 = all(P.labeledposMarked{ne},1);
    if ~all(l1==l2),
      fprintf('Marked and labels dont match for %d,%d\n',ndx,nexp);
    end
  end
  movf = [movf;K];
  lpos = [lpos; P.labeledpos];
  lmarked = [lmarked; P.labeledposMarked];
  
end

%%
J.movieFilesAll = movf;
J.labeledpos = lpos;
J.labeledposMarked = lmarked;
J.cfg = P.cfg;

%%

rfn = get_readframe_fcn(Q.vid1files{1});
ii = rfn(Q.ts(1));
figure; imshow(ii);
hold on;
scatter(Q.pts(1,1,:,1),Q.pts(2,1,:,1),'.');
hold off;

%%

pp = [];
for ndx = 1:numel(J.labeledpos)
  ff = ~isnan(J.labeledpos{ndx}(1,1,:));
  pp = cat(3,pp,J.labeledpos{ndx}(:,:,ff));
  
end


%%
f = figure;

nc = 5; nr = 4;
for ndx = 6:10
  for zz = 1
    subplot(nc,nr,ndx);
    rpt = squeeze(pp(8,zz,:));
    hist(squeeze(pp(ndx,zz,:))-rpt);
  end  
end


%% label for 318

kk = [];
for ndx = 1:numel(Q.expdirs)
%   if ~isempty(strfind(Q.expdirs{ndx},'138'));
%     kk(end+1) = ndx;
%   end
  fprintf('%s\n',Q.expdirs{ndx}(46:end));
end

