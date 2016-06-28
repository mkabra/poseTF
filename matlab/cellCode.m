%% Compute 3d tracking

addpath netlab
expdir = '/home/mayank/work/PoseEstimationData/Stephen/fly219/fly219_trial1/';
savefile = fullfile(expdir,'Out_3d.mat');
fvidfile = fullfile(expdir,'C002H001S0001','C002H001S0001_c.avi');
svidfile = fullfile(expdir,'C001H001S0001','C001H001S0001_c.avi');
fmat = fullfile(expdir,'C002H001S0001','projects__fly219_trial1__0001.mat');
smat = fullfile(expdir,'C001H001S0001','projects__fly219_trial1__0001_side.mat');
kinefile = fullfile(expdir,'01_fly219_kineData_kine.mat');
dosave = false;

compute3Dfrom2D(savefile,fvidfile,svidfile,fmat,smat,kinefile,dosave);

%% 3d back to 2d

ftrk = fullfile(expdir,'C002H001S0001','C002H001S0001_c.trk');
strk = fullfile(expdir,'C001H001S0001','C001H001S0001_c.trk');

convertResultsToTrx(savefile,ftrk,strk);