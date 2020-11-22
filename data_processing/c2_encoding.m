function mat = c2_encoding(subseq)
l = length(subseq);
mat = zeros(l,2);
for i = 1:l
    s = subseq(i);
    if strcmp(s,'A')
        mat(i,:) = [0,0];
    elseif strcmp(s,'C')
        mat(i,:) = [1,1];
    elseif strcmp(s,'G')
        mat(i,:) = [1,0];
    elseif strcmp(s,'T')
        mat(i,:) = [0,1];
    else
        mat(i,:) = [0,0];
    end
end
end
