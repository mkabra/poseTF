%% paths

% addpath /groups/branson/bransonlab/mayank/JAABA/filehandling;
% addpath /groups/branson/bransonlab/mayank/JAABA/misc;
addpath ~bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath ~bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;

addpath /groups/branson/bransonlab/projects/flyHeadTracking/code/
addpath ~bransonk/tracking/code/Ctrax/matlab/netlab

%% 

dd = dir('/groups/branson/home/kabram/bransonlab/PoseTF/results/headResults/movies/*_side.avi');

expnames = {};
for ndx = 5:numel(dd)
  ii = dd(ndx).name(1:end-9);
  [xx,~,~,fstr] = regexp(dd(ndx).name,'fly(_*\d+)');
  assert(~isempty(xx),'filename is weird');
  fparts = strsplit(dd(ndx).name(1:end-9),'__');
  experiment_name = ii;
  if strcmp(ii(1:10),'PoseEstima'),
    bdir = '/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/';
    
    frontviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),'C002H001S0001','C002H001S0001_c.avi');
    sideviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),'C001H001S0001','C001H001S0001_c.avi');
    frontviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.mat']);
    sideviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-4) '.mat']);
    kdd = dir(fullfile(bdir,fstr{1},'kineData','*.mat'));
    assert(numel(kdd)==1,sprintf('kinedata weird for %d %s',ndx,dd(ndx).name));
    kinematfile = fullfile(bdir,fstr{1},'kineData',kdd(1).name);
    frontviewresultsvideofile = fullfile('/groups/branson/home/kabram/bransonlab/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.avi']);
    trainingdatafile = '/groups/branson/bransonlab/projects/flyHeadTracking/CNNTrackingResults20160409/FlyHeadStephenTestData_20160318.mat';
    
  elseif strcmp(ii(1:10),'flyHeadTra')
    ff = fopen('/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/FlyNumber2CorrespondingDLTfile.csv','r');
    K = textscan(ff,'%d,%s');
    fclose(ff);
    flynum = str2double(fstr{1}(4:end));
    mndx = find(K{1}==flynum);
    if isempty(mndx),
      fprintf('%d:%s dont have kinedata .. skipping\n',ndx,dd(ndx).name);
      continue;
    end
    bdir = '/groups/branson/bransonlab/projects/flyHeadTracking/ExamplefliesWithNoTrainingData/';
    frontviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),'C002H001S0001','C002H001S0001_c.avi');
    sideviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),'C001H001S0001','C001H001S0001_c.avi');
    frontviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.mat']);
    sideviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-4) '.mat']);
    kinematfile = fullfile('/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/DLTs/',[K{2}{mndx} '_kine.mat']);
    frontviewresultsvideofile = fullfile('/groups/branson/home/kabram/bransonlab/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.avi']);
    trainingdatafile = '/groups/branson/bransonlab/projects/flyHeadTracking/CNNTrackingResults20160409/FlyHeadStephenTestData_20160318.mat';
    
  else
    bdir = ['/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen/' fparts{1} '/data/'];
    frontviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),'C002H001S0001','C002H001S0001.avi');
    sideviewvideofile = fullfile(bdir,fstr{1},fparts{2}(2:end),'C001H001S0001','C001H001S0001.avi');
    frontviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.mat']);
    sideviewmatfile = fullfile('/groups/branson/bransonlab/mayank/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-4) '.mat']);
    kdd = fullfile(bdir,'kineData',['kinedata_' fparts{2}(2:end) '.mat']);
    assert(exist(kdd,'file')>0,sprintf('kinedata weird for %d %s',ndx,dd(ndx).name));
    kinematfile = kdd;
    frontviewresultsvideofile = fullfile('/groups/branson/home/kabram/bransonlab/PoseTF/results/headResults/movies/',[dd(ndx).name(1:end-9) '.avi']);
    trainingdatafile = '/groups/branson/bransonlab/projects/flyHeadTracking/CNNTrackingResults20160409/FlyHeadStephenTestData_20160318.mat';
    
  end
  
  if ~exist(frontviewvideofile,'file'),
    fprintf('Didnt find front video file for %d %s\n',ndx,dd(ndx).name);
    continue;
  end
  if ~exist(sideviewvideofile,'file'),
    fprintf('Didnt find side view video file for %d %s\n',ndx,dd(ndx).name);
    continue;
  end
  if ~exist(frontviewmatfile,'file'),
    fprintf('Didnt find front view mat file for %d %s\n',ndx,dd(ndx).name);
    continue;
  end
  if ~exist(sideviewmatfile,'file'),
    fprintf('Didnt find side view mat file for %d %s\n',ndx,dd(ndx).name);
    continue;
  end
  if ~exist(kinematfile,'file'),
    fprintf('Didnt find kinemat file for %d %s\n',ndx,dd(ndx).name);
    continue;
  end
  if ~exist(frontviewresultsvideofile,'file'),
    fprintf('Didnt find frontviewresultsvideofile file for %d %s\n',ndx,dd(ndx).name);
    continue;
  end
  offx_front = -1;
  offy_front = -1;
  offx_side = -1;
  offy_side = -1;
  scale_front = 4;
  scale_side = 4;

  Script2DTo3DTracking20160409;
end
