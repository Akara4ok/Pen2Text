import React from 'react';
import classes from './TextEditor.scss';
import Button from '@Components/Button/Button';
import DropdownList from '@Components/DropdownList/DropdownList';
import Backdrop from '@Components/Backdrop/Backdrop';
import Spinner from '@Components/Spinner/Spinner';
import { Editor } from 'react-draft-wysiwyg';
import { EditorState, ContentState } from 'draft-js';
import 'react-draft-wysiwyg/dist/react-draft-wysiwyg.css';
import { convertToRaw } from 'draft-js';
import htmlToDraft from 'html-to-draftjs';
import draftToHtml from 'draftjs-to-html';
import { buildHtmlFromResponse, prepareHtmlForDoc, exportToWord, saveFile } from '../../utils/utils';
import jsPDF from 'jspdf';

class TextEditor extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            plainText: [],
            downloadType: '*.txt',
            editorState: EditorState.createEmpty(),
            isSaving: false,
        };
    }

    componentDidUpdate(_, prevState) {
        let plainText = this.props.plainText;
        if (
            !this.props.plainText.length ||
            JSON.stringify(plainText) === JSON.stringify(prevState.plainText)
        ) {
            return;
        }
        const htmlView = buildHtmlFromResponse(plainText).innerHTML;
        const contentBlock = htmlToDraft(htmlView);
        if (contentBlock) {
            const contentState = ContentState.createFromBlockArray(
                contentBlock.contentBlocks,
            );
            const editorState = EditorState.createWithContent(contentState);
            this.setState({ plainText: plainText, editorState: editorState });
        }
    }

    onEditorStateChange = editorState => {
        this.setState({
            editorState,
        });
    };

    onDownloadClick = () => {
        const { downloadType } = this.state;
        const filename = 'result';
        if (downloadType === '*.txt') {
            this.saveTxtFile(filename);
        } else if (downloadType === '*.docx') {
            this.saveDocFile(filename);
        } else if (downloadType === '*.pdf') {
            this.savePDFFile(filename);
        }
    };

    saveTxtFile = filename => {
        const { editorState } = this.state;
        const blocks = convertToRaw(editorState.getCurrentContent()).blocks;
        const value = blocks
            .map(block => (!block.text.trim() && '\n') || block.text)
            .join('\n');
        const file = new Blob([value], { type: 'text/plain' });
        const saveOpts = {
            suggestedName: filename + '.txt',
            types: [
                {
                    description: 'Text file',
                    accept: { 'text/plain': ['.txt'] },
                },
            ],
        };
        saveFile(file, saveOpts);
    };

    saveDocFile = filename => {
        const { editorState } = this.state;
        const rawContentState = convertToRaw(editorState.getCurrentContent());
        const markup = draftToHtml(rawContentState);
        const sanitizeHTML = prepareHtmlForDoc(markup);
        const file = exportToWord(sanitizeHTML);
        const saveOpts = {
            suggestedName: filename + '.docx',
            types: [
                {
                    description: 'Doc file',
                    accept: { 'application/msword': ['.docx'] },
                },
            ],
        };
        saveFile(file, saveOpts);
    };

    savePDFFile = filename => {
        const { editorState } = this.state;
        const rawContentState = convertToRaw(editorState.getCurrentContent());
        const markup = draftToHtml(rawContentState);
        const doc = new jsPDF();

        const separator = '<p>' + '-'.repeat(50) + '</p>\n';
        const pages = markup.split(separator);
        console.log(pages);

        let index = 0;

        const saveCalback = doc => {
            const blobPDF = new Blob([doc.output('blob')], {
                type: 'application/pdf',
            });
            saveFile(blobPDF, saveOpts);
            this.setState({ isSaving: false });
        };

        const height = doc.internal.pageSize.getHeight();

        const createPageCallback = doc => {
            let element = pages[index];
            const pageCount = doc.internal.getNumberOfPages() - 1;
            console.log(pageCount);
            doc.html(element, {
                callback:
                    index === pages.length - 1
                        ? saveCalback
                        : createPageCallback,
                x: 0,
                y: 0 + (height - 30) * pageCount,
                width: 170,
                windowWidth: 650,
                margin: [15, 15],
                autoPaging: 'text',
            });
            if (index < pages.length - 1) {
                doc.addPage();
                index++;
            }
        };

        const saveOpts = {
            suggestedName: filename + '.pdf',
            types: [
                {
                    description: 'PDF file',
                    accept: { 'application/pdf': ['.pdf'] },
                },
            ],
        };

        this.setState({ isSaving: true });
        createPageCallback(doc);
    };

    setValue = type => {
        this.setState({ downloadType: type });
    };

    render() {
        const { editorState, isSaving } = this.state;
        return (
            <div className={classes.wrapper}>
                <Editor
                    editorState={editorState}
                    toolbarClassName={classes.toolbarClassName}
                    wrapperClassName={classes.wrapperClassName}
                    editorClassName={classes.editorClassName}
                    onEditorStateChange={this.onEditorStateChange}
                />
                <DropdownList
                    className={classes.dropdownStyle}
                    setValue={this.setValue}
                    items={['*.txt', '*.docx', '*.pdf']}
                />
                <Button
                    className={classes.buttonStyle}
                    onClick={this.onDownloadClick}>
                    Download
                </Button>
                {isSaving ? (
                    <Backdrop>
                        <Spinner />
                    </Backdrop>
                ) : null}
            </div>
        );
    }
}

export default TextEditor;
