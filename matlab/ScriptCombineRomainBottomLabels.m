l1 = load('/home/mayank/work/poseEstimation/RomainLeg/Apr28AndJun22_onlyBottom.lbl','-mat');
lfiles = {'/home/mayank/Dropbox/MultiViewFlyLegTracking/sep1616/sep1616-1531Romain.lbl'
  '/home/mayank/Dropbox/MultiViewFlyLegTracking/sep1516/sep1516-1537Romain.lbl'
  '/home/mayank/Dropbox/MultiViewFlyLegTracking/sep1316/sep1316-1606Romain.lbl'};

l = l1;
l.movieFilesAll{2} = '/home/mayank/Dropbox/MultiViewFlyLegTracking/older stuff/trackingApril28-14-53/bias_video_cam_2_date_2016_04_28_time_14_53_16_v001.avi';
l.movieFilesAll{3} = '/home/mayank/Dropbox/MultiViewFlyLegTracking/older stuff/trackingApril28-15-23/bias_video_cam_2_date_2016_04_28_time_15_23_20_v001.avi';

for ndx = 1:numel(lfiles),
  l2 = load(lfiles{ndx},'-mat');

  l.movieFilesAll{end+1} = ['/home/mayank/Dropbox/' l2.movieFilesAll{3}(33:end)];
  l.labeledpos{end+1} = l2.labeledpos{1}(37:end,:,:);
end
save('/home/mayank/work/poseEstimation/RomainLeg/Apr28Jun22Sep16Sep15Sep13_onlyBottom.lbl','-struct','l','-v7.3');

