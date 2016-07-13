function GMMTrack2DTo3D(fmoviefile,smoviefile,kinelistfile,outdir,redo)

%% Constructs smoothed 3D construction for videos listed in file

addpath ~bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath ~bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;

addpath /groups/branson/bransonlab/projects/flyHeadTracking/code/
addpath ~bransonk/tracking/code/Ctrax/matlab/netlab

% infile = '~/bransonlab/PoseEstimationData/Stephen/folders2track.txt';
% outdir = '/nobackup/branson/mayank/stephenOut/';
% kinelistfile = '/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/FlyNumber2CorrespondingDLTfile.csv';
% kineDir = '/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/DLTs/';

if nargin < 5,
  redo = false;
end
%% read the list of files
fidf = fopen(fmoviefile,'r');
fids = fopen(smoviefile,'r');

Xf = textscan(fidf,'%s');
Xf = Xf{1};
Xs = textscan(fids,'%s');
Xs = Xs{1};
fclose(fidf);fclose(fids);

assert(numel(Xf)==numel(Xs),'Number of movies in front view file and side view text file should be same');

ff = fopen(kinelistfile,'r');
K = textscan(ff,'%d,%s');
fclose(ff);

%%

parfor ndx = 1:numel(Xf)
  fparts = strsplit(Xf{ndx},filesep);
  outf = sprintf('%s__%s__%s_front.mat',fparts{end-4},fparts{end-1},fparts{end}(end-3:end));
  fparts = strsplit(Xs{ndx},filesep);
  outs = sprintf('%s__%s__%s_side.mat',fparts{end-4},fparts{end-1},fparts{end}(end-3:end));
  matfilef = fullfile(outdir,outf);
  matfiles = fullfile(outdir,outs);
  if ~exist(matfilef,'file') || ~exist(matfiles,'file'),
    fprintf('Outfile %s or %s dont exist for %s\n',matfilef,matfiles,Xf{ndx})
    continue;
  end
  flynum = str2double(fparts{end-2}(4:6));
  mndx = find(K{1}==flynum);
  if isempty(mndx),
    fprintf('%d:%s dont have kinedata .. skipping\n',ndx,Xf{ndx});
    continue;
  end
  
  kinematfile = fullfile(kineDir,[K{2}{mndx} '_kine.mat']);
  outfile = [Xf{ndx} '_3Dres'];
  
  if exist([outfile '.mat'],'file') && ~redo,
    continue
  end
  
  if ndx<0,
    makevideo = true;
  else
    makevideo = false;
  end
  [~,dname] = fileparts(Xf{ndx});
  fmov = fullfile(Xf{ndx},[dname '_c.avi']);
  [~,dname] = fileparts(Xs{ndx});
  smov = fullfile(Xs{ndx},[dname '_c.avi']);
  fprintf('Working on %d:%s\n',ndx,Xf{ndx});
  compute3Dfrom2D(outfile,fmov,smov,matfilef,matfiles,kinematfile,makevideo);
  fprintf('Done.\n');

  % convert back to 2D
    matfile = [Xf{ndx} '_3Dres.mat'];
  if ~exist(matfile,'file'),
    fprintf('3D mat file dont exist for %s\n',Xf{ndx})
    continue;
  end
  [~,fname] = fileparts(Xf{ndx});
  ftrx = fullfile(Xf{ndx}, [fname '_c.trk']);
  
  [~,sname] = fileparts(Xs{ndx});
  strx = fullfile(Xs{ndx}, [sname '_c.trk']);
  
  convertResultsToTrx(matfile,ftrx,strx);
  fprintf('Done conversion for %d %s\n',ndx,Xf{ndx});

  
end
