from matplotlib.backends.backend_pdf import PdfPages
import model as CMEmodel
import reporter as CMEreporter


# Driver for CME cost model.

if __name__ == "__main__":

    cm = CMEmodel.CMECostModel(zero_intcpt=True, spread_term=True)

    # pdf = PdfPages('../../figs/model_diagnostic_default.pdf')
    pdf = PdfPages('../../figs/fig2.pdf')
    r = CMEreporter.Reporter('.',
                             'All',
                             qc=True,
                             pdf=pdf,
                             closeonExit=True)
    cm.accept(r)
