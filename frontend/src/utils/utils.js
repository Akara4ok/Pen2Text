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

export const prepareHtmlForDoc = htmlString => {
    let content = htmlString;
    let currentPos = 0;
    const separator = '<p>' + '-'.repeat(50) + '</p>';
    while (content.indexOf(separator, currentPos) !== -1) {
        currentPos = content.indexOf(separator, currentPos) + separator.length;
        content =
            content.slice(0, currentPos) +
            ' <br style="page-break-before: always"> ' +
            content.slice(currentPos);
    }
    content = content.replaceAll(separator, '');
    return content;
};

export const exportToWord = element => {
    const preHtml =
        '<html' +
        "xmlns:o='urn:schemas-microsoft-com:office:office'" +
        "xmlns:w='urn:schemas-microsoft-com:office:word'" +
        "xmlns='http://www.w3.org/TR/REC-html40'>" +
        '<head>' +
        "<meta charset='utf-8'>" +
        '<title>Export HTML To Doc</title>' +
        '</head>' +
        '<body>';
    const postHtml = '</body></html>';
    const html = preHtml + element + postHtml;
    const blob = new Blob(['\ufeff', html], {
        type: 'application/msword',
    });

    return blob;
};

export const saveFile = (content, opts) => {
    window
        .showSaveFilePicker(opts)
        .then(fileHandler => {
            fileHandler.createWritable().then(writable => {
                writable.write(content).then(() => {
                    writable.close();
                });
            });
        })
        .catch(() => {});
};
