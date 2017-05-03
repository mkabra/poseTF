function compile_compute3Dfrom2D()

% get the directory where this file lives
% figure out where the root of the Ohayon code is
thisScriptFileName=mfilename('fullpath');
thisScriptDirName=fileparts(thisScriptFileName);

% just put the executable in the did with the build script
exeDirName=thisScriptDirName;


mcc('-o','compute3Dfrom2D_compiled', ...
    '-m', ...
    '-d',fullfile(exeDirName,'compiled'), ...
    '-I','/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling',...
    '-I','/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc',...
    '-I','/groups/branson/bransonlab/projects/flyHeadTracking/code/',...
    '-I','/groups/branson/home/bransonk/tracking/code/Ctrax/matlab/netlab',...
    '-I','/groups/branson/bransonlab/mayank/APT/user/orthocam',...
    '-I','/groups/branson/bransonlab/mayank/APT/misc',...
    '-a','/groups/branson/bransonlab/mayank/APT/user/orthocam/OrthoCam.m',...
    '-a','/groups/branson/bransonlab/mayank/APT/user/orthocam/OrthoCamCalPair.m',...
    '-v', ...
    '-R','-singleCompThread',...
    fullfile(exeDirName,'compute3Dfrom2D_KB.m'));

