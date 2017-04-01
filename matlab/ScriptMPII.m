%%
A = load('/groups/branson/home/kabram/bransonlab/pose_hourglass/pose_hg_demo/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat');
A = A.RELEASE;
%%

imdir = '/groups/branson/bransonlab/mayank/pose_hourglass/pose_hg_demo/images';
id = 1;
imfile = fullfile(imdir,A.annolist(ndx).image.name);
im = imread(imfile);

%%
figure; 
imshow(im);
pts = A.annolist(ndx).annorect(1).annopoints;
hold on;
scatter([pts.point(:).x],[pts.point(:).y],'.');

%%

zz = zeros(0,16,2);
for ndx = 1:numel(A.annolist)
  if ~A.img_train(ndx), continue; end
%   if ~isfield(A.annolist(ndx).annorect,'annopoints'),
%     continue;
%   end
  if isempty(A.annolist(ndx).annorect), continue; end
  if ~isfield(A.annolist(ndx).annorect,'scale')
    continue;
  end
  if ~isfield(A.annolist(ndx).annorect,'annopoints')
    continue;
  end
  for pt = 1:numel(A.annolist(ndx).annorect)
    scale = A.annolist(ndx).annorect(pt).scale;
    if isempty(scale), continue; end
    if isempty(A.annolist(ndx).annorect(pt).annopoints), continue; end
    cc = nan(16,2);
    for dd = 1:numel(A.annolist(ndx).annorect(pt).annopoints.point)
      curpt = A.annolist(ndx).annorect(pt).annopoints.point(dd);
      cc(curpt.id+1,1) = curpt.x;
      cc(curpt.id+1,2) = curpt.y;
    end
    cc = cc/scale;
    zz(end+1,:,:) = cc;
  end
end

%%