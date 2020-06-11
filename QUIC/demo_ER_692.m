% $Id: demo_ER_692.m,v 1.8 2012-05-11 00:40:35 sustik Exp $
% Demontrates the usage of QUIC().

% Contains the 692 x 692 empirical covariance matrix: 
load('ER_692.mat');
% Obtained from: http://www.math.nus.edu.sg/~mattohkc/Covsel-0.zip
% See also: Lu Li, Kim-Chuan Toh: An inexact interior point method for L1
% regularized sparse covariance selection.  Math. Prog. Comp. (2010)
% 2:291-315

% Run in "default" mode on the ER_692 dataset:
[X W opt cputime iter dGap] = QUIC('default', S, 0.5, 1e-6, 2, 100);

% Run in path mode on the ER_692 dataset:
[XP WP optP cputimeP iterP dGapP] = QUIC('path', S, 1.0, ...
                                         [1.0 0.9 0.8 0.7 0.6 0.5], ...
                                         1e-16, 2, 100);

% Run in "trace" mode on the ER_692 dataset:
[XT WT optT cputimeT iterT dGapT] = QUIC('trace', S, 0.5, 1e-16, 1, iter);

% opt = 923.104246042393
