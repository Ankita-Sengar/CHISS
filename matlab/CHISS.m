function [syntheticData, balancedData, syntheticLabels, stats] = CHISS(X, y, minorityLabel, varargin)
%CHISS  Clean Hyperspherical Inter-Sample Similarity oversampling (Algorithm 1).
%
%   [Xsyn, balanced, ysyn, stats] = CHISS(X, y, minorityLabel)
%   [...] = CHISS(X, y, minorityLabel, 'Name', value, ...)
%
%   X              n x d numeric matrix (scale it first, e.g. min-max on the training data)
%   y              n x 1 label vector (binary)
%   minorityLabel  label of the minority class
%
%   Name-value options (defaults = paper settings, Table 2):
%     'K0'          9        upper bound of the global neighbourhood size (Eq. 5)
%     'Khat0'       7        upper bound of the local neighbourhood size (Eq. 7)
%     'a'           10       slope of the sigmoids in Eqs. (5) and (7)
%     'b'           0.01     offset of the sigmoids in Eqs. (5) and (7)
%     'nMajMax'     1        max. number of majority samples inside an admissible hypersphere
%     'tauOverlap'  0.5      overlap check: accept if d_maj / d_min >= tauOverlap
%     'deltaDup'    0.005    duplication check: accept if distance to nearest sample >= deltaDup
%     'distance'    'l1'     'l1' or 'l2'
%     'target'      'default' 'default' (Eq. 3), 'full' (theta = 1) or 'conservative'
%
%   Each minority anchor makes ceil(nSyn/nMin) sampling attempts using
%   admissible radii selected with replacement. Candidates failing the
%   overlap or duplication checks are discarded. The total accepted
%   count may be less than nSyn. Set rng(seed) for reproducible runs.
%
%   Outputs
%     syntheticData    m x d synthetic minority samples (m <= n_syn)
%     balancedData     [X; syntheticData] with labels in the last column
%     syntheticLabels  m x 1 (all minorityLabel)
%     stats            requested / accepted samples, K_D, n_ind, active anchors, rejections

    p = inputParser;
    p.addParameter('K0', 9);          p.addParameter('Khat0', 7);
    p.addParameter('a', 10);          p.addParameter('b', 0.01);
    p.addParameter('nMajMax', 1);     p.addParameter('tauOverlap', 0.5);
    p.addParameter('deltaDup', 0.005);
    p.addParameter('distance', 'l1'); p.addParameter('target', 'default');
    p.parse(varargin{:}); o = p.Results;

    X = double(X); y = double(y(:)); minorityLabel = double(minorityLabel);
    switch lower(o.distance)
        case 'l1', distName = 'cityblock';
        case 'l2', distName = 'euclidean';
        otherwise, error('CHISS:distance', 'distance must be ''l1'' or ''l2''.');
    end

    isMin = (y == minorityLabel);
    Xmin = X(isMin, :); Xmaj = X(~isMin, :);
    nMin = size(Xmin, 1); nMaj = size(Xmaj, 1); d = size(X, 2);

    syntheticData = zeros(0, d); syntheticLabels = zeros(0, 1); balancedData = [X, y];
    stats = struct('requested', 0, 'accepted', 0, 'K_D', 0, 'n_ind', 0, 'activeAnchors', 0, ...
                   'admissibleAnchors', 0, 'rejectedOverlap', 0, 'rejectedDuplicate', 0, 'nMin', nMin);
    if nMin < 2 || nMaj == 0 || nMin >= nMaj, return; end

    %% Line 2: Eqs. (2)-(5)
    DeltaG = nMin / nMaj;                                         % Eq. (2)
    switch lower(o.target)                                        % Eq. (3)
        case 'default'
            if DeltaG < 0.01, theta = 0.7; elseif DeltaG < 0.2, theta = 1.0; else, theta = max(0.9, DeltaG); end
        case 'full'
            theta = 1.0;
        case 'conservative'
            if DeltaG < 0.01, theta = 0.5; elseif DeltaG < 0.2, theta = 0.9; else, theta = max(0.85, DeltaG); end
        otherwise
            error('CHISS:target', 'Unknown target ''%s''.', o.target);
    end
    nSyn = max(0, round(theta * nMaj - nMin));                    % Eq. (4)
    KD = ceil(o.K0 / (1 + exp(o.a * (DeltaG - o.b))));            % Eq. (5)
    KD = min(max(KD, 1), size(X, 1) - 1);
    stats.requested = nSyn; stats.K_D = KD;
    if nSyn == 0, return; end

    %% Line 3: per-anchor quota (= number of attempts per anchor)
    nInd = ceil(nSyn / nMin); stats.n_ind = nInd;

    [nbrIdx, nbrDist] = knnsearch(X, Xmin, 'K', KD + 1, 'Distance', distName);
    minIdx = find(isMin);
    NSmin = createns(Xmin, 'Distance', distName);
    NSmaj = createns(Xmaj, 'Distance', distName);

    syn = zeros(nSyn, d); nGood = 0; nOv = 0; nDup = 0; nActive = 0; nAdm = 0;

    for i = 1:nMin                                                % Line 5
        idx = nbrIdx(i, :); dst = nbrDist(i, :);
        self = find(idx == minIdx(i), 1); if isempty(self), self = numel(idx); end
        idx(self) = []; dst(self) = [];
        idx = idx(1:KD); dst = dst(1:KD);

        lab = y(idx);                                             % Line 6: Eq. (6)
        DeltaL = sum(lab == minorityLabel) / max(sum(lab ~= minorityLabel), eps);
        KL = ceil(o.Khat0 / (1 + exp(-o.a * (DeltaL - o.b))));    % Line 7: Eq. (7)
        KF = max(1, min(KD, KL));                                 % Line 8
        Zd = dst(1:KF); Zl = lab(1:KF);                           % Line 9

        majInside = cumsum(Zl ~= minorityLabel);                  % Lines 10-16: admissible radii Q_i
        Q = Zd(majInside(:) <= o.nMajMax & Zd(:) > 0); Q = Q(:);
        if ~isempty(Q), nAdm = nAdm + 1; end
        if nGood >= nSyn || isempty(Q), continue; end
        nActive = nActive + 1;

        xc = Xmin(i, :);
        for t = 1:nInd                                            % Line 19: n_ind attempts (Line 18: Q_i non-empty)
            if nGood >= nSyn, break; end
            rc = Q(randi(numel(Q)));                              % Line 20: sample radius with replacement
            xs = xc + rc * rand()^(1 / d) * unitDirection(d, o.distance);
            [~, dMin] = knnsearch(NSmin, xs);
            [~, dMaj] = knnsearch(NSmaj, xs);
            if dMaj / (dMin + eps) >= o.tauOverlap && min(dMin, dMaj) >= o.deltaDup
                nGood = nGood + 1; syn(nGood, :) = xs;
            elseif min(dMin, dMaj) < o.deltaDup
                nDup = nDup + 1;
            else
                nOv = nOv + 1;
            end
        end
    end

    syntheticData = syn(1:nGood, :);
    syntheticLabels = repmat(minorityLabel, nGood, 1);
    balancedData = [[X; syntheticData], [y; syntheticLabels]];
    stats.accepted = nGood; stats.activeAnchors = nActive; stats.admissibleAnchors = nAdm;
    stats.rejectedOverlap = nOv; stats.rejectedDuplicate = nDup;
end

function v = unitDirection(d, mode)
% L1: normalised unit-rate exponentials with random signs (uniform on the unit L1 sphere).
    if strcmpi(mode, 'l1')
        e = -log(rand(1, d));
        v = (2 * (rand(1, d) > 0.5) - 1) .* (e / sum(e));
    else
        g = randn(1, d); v = g / max(norm(g), eps);
    end
end
