# DerivPort interface

Design workspace: [Hackathon — DerivPort](https://www.figma.com/design/ZxLybASxv0lxyUkdnQ33A2/Hackathon-DerivPort?node-id=1-7), in the separate Hackathon Figma team. The **Reference screens** page contains the completed research and algorithm-builder frames with editable text, shapes, and chart paths. File access follows its Figma sharing settings.

The portable vector sources are [research.svg](../design/research.svg) and [algorithm-builder.svg](../design/algorithm-builder.svg). Paste their SVG contents onto a Figma canvas to import editable layers. The earlier native auto-layout studies remain on Page 1.

The interface uses the structure of an equity research desk: large price typography, an immediate historical chart, dated company evidence, and a distinct quantitative model panel. Rounded tabs and actions provide the requested bubble treatment without repeating the same floating card around every section.

| Role | Color |
| --- | --- |
| Canvas | `#F3F1EB` |
| Ink and market chart | `#101820` |
| Model and primary actions | `#2454FF` |
| Historical price line | `#D6F23C` |

Small decorative labels above headings are omitted. The app uses local system sans-serif and monospace fonts, keeping the offline presentation independent of font services. LaTeX equations retain the application's math rendering.

Research shows the last 126 observed sessions through the chosen cutoff. The full available history remains in an expandable view. Both charts slice the series at the cutoff before plotting, so the redesign does not reveal future prices. The cache date, company-source staleness, pending outcomes, and simulated nature of future scenarios remain visible.

The Figma layout illustrates a selected balanced algorithm with `k = 1`. The app continues to start with its existing custom expression and `k = 2`; selecting **Use my balanced algorithm** reproduces that design state. Figma is a design reference; the running app computes scores, rankings, derivatives, and backtests from the selected data.
