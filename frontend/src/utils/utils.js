export const srcToFile = (src, fileName, mimeType) => {
    return fetch(src)
        .then(function (res) {
            return res.arrayBuffer();
        })
        .then(function (buf) {
            return new File([buf], fileName, { type: mimeType });
        });
};

export const buildHtmlFromResponse = texts => {
    const content = document.createElement('div');
    if (!texts) return content;
    for (let index = 0; index < texts.length; index++) {
        const text = texts[index];
        const page = document.createElement('p');
        page.innerText = text;
        content.appendChild(page);
        if (index !== texts.length - 1) {
            const separator = document.createElement('p');
            separator.innerText = '-'.repeat(50);
            content.appendChild(separator);
        }
    }
    return content;
};