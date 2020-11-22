function [ mat ] = codon64( seq )
c64 = codoncount(seq);
mat = zeros(64,1);
z = fieldnames(c64);
for i = 1:64
    mat(i) = c64.(z{i});
end
end

% function [ mat ] = codon64( seq )
% c64 = codoncount(seq);
% mat = struct2cell(c64);
% mat = cell2mat(mat);
% end
% 


