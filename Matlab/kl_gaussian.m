function kl_gaussian(kl)
    d = 2*(kl^0.5);
    extra = 5;
    xvals = -extra:0.02:(d+extra);
    pdf1 = normpdf(xvals, 0, 1);
    pdf2 = normpdf(xvals, d, 1);
    figure;
    plot(xvals, pdf1)
    hold on
    plot(xvals, pdf2)
    xlabel('$x$')
    legend("$p_1(x)$", "$p_2(x)$")
end