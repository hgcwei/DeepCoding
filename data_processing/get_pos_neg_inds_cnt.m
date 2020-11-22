function [inds_p,inds_n,is_err] = get_pos_neg_inds_cnt(len,cds,ratio)
is_err = 0;
if mod(length(cds),2) ~= 0
    is_err = 1;
    inds_p = [];
    inds_n = [];
    return;
end
pos_inds = get_pos_inds(cds);
neg_inds = setdiff(1:len,pos_inds);
lp = length(pos_inds);
ln = length(neg_inds);
num = ceil(min([lp,ln])*ratio);
rp = randperm(lp);
rn = randperm(ln);
inds_p = pos_inds(rp(1:num));
inds_n = neg_inds(rn(1:num));
end

function ind_p = get_pos_inds(cds)
inds_c = [];
for i = 1:2:length(cds)
    inds_c = [inds_c,cds(i):cds(i+1)];
end
ind_p = inds_c(1:3:length(inds_c));
end

