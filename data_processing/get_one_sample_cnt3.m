function mat = get_one_sample_cnt3(seq,ind,lws,rws,v)
l = length(seq);
s1 = blanks(lws);
s2 = blanks(rws);
s1(:)= 'N';
s2(:)='N';
seq_enl = [s1,seq,s2];

fst = ind - v;
sec = ind;
thr = ind+ v;

if fst < 1
    fst = 1;
end

if thr > l
    thr = l;
end
mat1 = hb_encoding(seq_enl(fst:fst+lws+rws));
mat2 = hb_encoding(seq_enl(sec:sec+lws+rws));
mat3 = hb_encoding(seq_enl(thr:thr+lws+rws));
mat = [mat1,mat2,mat3];
end


% function mat = e1_encoding(subseq)
% l = length(subseq);
% mat = zeros(l,1);
% for i = 1:l
%     s = subseq(i);
%     if strcmp(s,'A')
%         mat(i,:) = 2;
%     elseif strcmp(s,'C')
%         mat(i,:) = -1;
%     elseif strcmp(s,'G')
%         mat(i,:) = 1;
%     elseif strcmp(s,'T')
%         mat(i,:) = -2;
%     else
%         mat(i,:) = 0;
%     end
% end
% end

