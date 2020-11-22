clc
clear;
load train_seqs15000cell
load train_cds15000cell

load test_seqs4900cell
load test_cds4900cell

% train_pos_path = 'train1000coding/1/';
% train_neg_path = 'train1000coding/0/';
% test_pos_path = 'test500coding/1/';
% test_neg_path = 'test500coding/0/';

train_pos_path2 = 'train/1/';
train_neg_path2 = 'train/0/';
test_pos_path2 = 'test/1/';
test_neg_path2 = 'test/0/';

lws = 45;
rws = 44;
ratio = 0.05;
v = 90;
for i = 1:15000
    i
    seq = upper(train_seqs15000cell{i});
    cds = train_cds15000cell{i};
    l = length(seq);
    [pind,nind,is_err] = get_pos_neg_inds_cnt(l,cds,ratio);
    if ~is_err
        for j = 1:length(pind)
            mat = get_one_sample_cnt3(seq,pind(j),lws,rws,v);
%             mat2 = get_one_sample_cnt(seq,pind(j),lws,rws);
%             file = strcat(train_pos_path,'seq-',num2str(i),'-loc-',num2str(pind(j)),'.png');
            file3 = strcat(train_pos_path2,'seq-',num2str(i),'-loc-',num2str(pind(j)),'.csv');
            dlmwrite(file3,mat);
%             imwrite(mat2gray(mat),file);
%             imwrite(mat2gray(mat2),file2);
        end

        for j = 1:length(nind)
%             mat = get_one_sample_cnt2(seq,nind(j),lws,rws);
            mat = get_one_sample_cnt3(seq,nind(j),lws,rws,v);
%             file = strcat(train_neg_path,'seq-',num2str(i),'-loc-',num2str(nind(j)),'.png');
            file2 = strcat(train_neg_path2,'seq-',num2str(i),'-loc-',num2str(nind(j)),'.csv');
            dlmwrite(file2,mat);
%             imwrite(mat2gray(mat),file);
%             imwrite(mat2gray(mat2),file2);
        end
    end
end

for i = 1:4900
    i
    seq = upper(test_seqs4900cell{i});
    cds = test_cds4900cell{i};
    l = length(seq);
    [pind,nind,is_err] = get_pos_neg_inds_cnt(l,cds,ratio);
    if ~is_err
        for j = 1:length(pind)
            mat = get_one_sample_cnt3(seq,pind(j),lws,rws,v);
%             mat2 = get_one_sample_cnt(seq,pind(j),lws,rws);
%             file = strcat(test_pos_path,'seq-',num2str(i),'-loc-',num2str(pind(j)),'.png');
            file2 = strcat(test_pos_path2,'seq-',num2str(i),'-loc-',num2str(nind(j)),'.csv');
%             imwrite(mat2gray(mat),file);
%             imwrite(mat2gray(mat2),file2);
            dlmwrite(file2,mat);
        end

        for j = 1:length(nind)
%             mat = get_one_sample_cnt2(seq,nind(j),lws,rws);
            mat = get_one_sample_cnt3(seq,nind(j),lws,rws,v);
%             file = strcat(test_neg_path,'seq-',num2str(i),'-loc-',num2str(nind(j)),'.png');
            file2 = strcat(test_neg_path2,'seq-',num2str(i),'-loc-',num2str(nind(j)),'.csv');
%             imwrite(mat2gray(mat),file);
%             imwrite(mat2gray(mat),file2);
            dlmwrite(file2,mat);
        end
    end
end