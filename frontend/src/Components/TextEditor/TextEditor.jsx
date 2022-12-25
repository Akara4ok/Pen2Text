import React from 'react';
import classes from './TextEditor.scss';
import Button from '@Components/Button/Button';
import DropdownList from '@Components/DropdownList/DropdownList';
import { Editor } from 'react-draft-wysiwyg';
import { EditorState, ContentState } from 'draft-js';
import 'react-draft-wysiwyg/dist/react-draft-wysiwyg.css';
import { convertToRaw } from 'draft-js';
import htmlToDraft from 'html-to-draftjs';
import { buildHtmlFromResponse } from '../../utils/utils';

class TextEditor extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            plainText: [],
            downloadType: '*.txt',
            editorState: EditorState.createEmpty(),
        };
    }

    componentDidUpdate(prevProps, prevState) {
        let { plainText } = this.state;
        plainText = this.props.plainText;
        if (
            !this.props.plainText.length ||
            JSON.stringify(plainText) === JSON.stringify(prevState.plainText)
        ) {
            return;
        }
        const htmlView = buildHtmlFromResponse(plainText).innerHTML ;
        console.log(htmlView);
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

    setValue = (type) => {
        this.setState({downloadType: type});
    }

    render() {
        const { editorState } = this.state;
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
            </div>
        );
    }
}

export default TextEditor;
