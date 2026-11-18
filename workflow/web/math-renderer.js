/* Typeset explicitly marked research prose without interpreting model HTML. */
(() => {
  "use strict";
  const ignored = "script,noscript,style,textarea,pre,code,input,select,option,.code-block,.katex,.katex-display";
  const environments = ["equation", "equation*", "align", "align*", "aligned", "alignat", "alignat*", "gather", "gather*", "gathered", "CD", "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "cases"];
  const delimiters = [
    {left: "$$", right: "$$", display: true},
    {left: "\\[", right: "\\]", display: true},
    {left: "\\(", right: "\\)", display: false},
    ...environments.map(name => ({left: `\\begin{${name}}`, right: `\\end{${name}}`, display: true})),
    {left: "$", right: "$", display: false},
  ];

  function standaloneLatex(line) {
    // Only standalone TeX expressions, never guess where math starts in prose.
    if (/\$|\\[()[\]]|\\(?:begin|end)\b/.test(line)) return false;
    if (!/\\(?:frac|dfrac|tfrac|sum|prod|int|sqrt|Delta|mu|lambda|theta|alpha|beta|gamma|omega|phi|psi|ell|mathbb|mathcal|langle|lVert|left|lim)\b/.test(line)) return false;
    const nonCommands = line.replace(/\\(?:text|operatorname|mathrm|mathsf)\{[^{}]*\}/g, "")
      .replace(/\\[a-zA-Z]+/g, "");
    return !/[a-zA-Z]{2,}/.test(nonCommands);
  }

  function prepareText(element) {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, {
      acceptNode: node => node.parentElement.closest(ignored) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const source = node.textContent;
      // Explicit math fences are formulas; ordinary fenced/inline code stays literal.
      const codeTokens = /```([^\n`]*)\n([\s\S]*?)```|`([^`\n]*)`/g;
      const fragment = document.createDocumentFragment();
      let cursor = 0;
      const appendText = text => {
        if (/\$|\\[()[\]]|\\(?:begin|end)\b/.test(text)) {
          fragment.append(document.createTextNode(text));
          return; // Never wrap lines inside an existing multi-line formula.
        }
        const lines = text.split("\n");
        for (let i = 0; i < lines.length; i++) {
          const line = lines[i];
          fragment.append(document.createTextNode(standaloneLatex(line) ? `$$${line.trim()}$$` : line));
          if (i < lines.length - 1) fragment.append(document.createTextNode("\n"));
        }
      };
      for (const match of source.matchAll(codeTokens)) {
        appendText(source.slice(cursor, match.index));
        const fenced = match[2] !== undefined;
        const mathFence = fenced && ["math", "latex", "tex"].includes(match[1].trim().toLowerCase());
        const part = document.createElement(mathFence ? "span" : fenced ? "pre" : "code");
        part.className = mathFence ? "math-fenced" : "math-code";
        part.textContent = mathFence ? `$$\n${match[2].trim()}\n$$` : fenced ? match[2] : match[3];
        fragment.append(part);
        cursor = match.index + match[0].length;
      }
      appendText(source.slice(cursor));
      node.replaceWith(fragment);
    }
  }

  function render(root) {
    if (!root) return;
    const regions = [...(root.matches?.("[data-math]") ? [root] : []), ...root.querySelectorAll("[data-math]")];
    for (const element of regions) {
      if (element.dataset.mathRendered || element.closest(ignored)) continue;
      if (typeof window.renderMathInElement !== "function") {
        element.dataset.mathUnavailable = "true";
        continue; // Plain text remains available if a local asset fails to load.
      }
      const original = element.textContent;
      if (original.length > 100000) continue;
      try {
        prepareText(element);
        window.renderMathInElement(element, {
          delimiters,
          ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "input", "select", "option"],
          ignoredClasses: ["code-block", "katex", "katex-display"],
          output: "htmlAndMathml",
          throwOnError: false,
          errorColor: "#906838",
          strict: "ignore",
          trust: false,
          maxSize: 8,
          maxExpand: 1000,
          macros: {},
          errorCallback: () => { element.dataset.mathError = "true"; },
        });
        element.querySelectorAll(".katex-display").forEach(formula => {
          formula.tabIndex = 0;
          formula.setAttribute("role", "group");
          formula.setAttribute("aria-label", "Display equation; scroll horizontally if needed");
        });
      } catch {
        element.textContent = original;
        element.dataset.mathError = "true";
      }
      element.dataset.mathRendered = "true";
      delete element.dataset.mathUnavailable;
    }
  }
  window.MathRenderer = Object.freeze({render});
})();
