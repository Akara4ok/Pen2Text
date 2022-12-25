import React from 'react';
import classes from './Layout.scss';
import FileUploader from '@Components/FileUploader/FileUploader';
import TextEditor from '@Components/TextEditor/TextEditor';
import Button from '@Components/Button/Button';

let lastTargetEvent;

class Layout extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isDragEnter: false,
        };
    }

    dragEnterHandler = event => {
        event.preventDefault();
        event.stopPropagation();
        const { isDragEnter } = this.state;
        lastTargetEvent = event.target;
        if (!isDragEnter) this.setState({ isDragEnter: true });
    };

    dragEndHandler = event => {
        const { isDragEnter } = this.state;
        if (isDragEnter && lastTargetEvent == event.target) {
            event.preventDefault();
            event.stopPropagation();
            this.setState({ isDragEnter: false });
        }
    };

    onDropHandler = event => {
        const { isDragEnter } = this.state;
        event.preventDefault();
        event.stopPropagation();
        if (isDragEnter) {
            lastTargetEvent = undefined;
            this.setState({ isDragEnter: false });
        }
    };

    dragOverHandler = event => {
        event.preventDefault();
    };

    render() {
        const { isDragEnter } = this.state;
        return (
            <div
                className={classes.wrapper}
                onDragEnter={this.dragEnterHandler}
                onDragLeave={this.dragEndHandler}
                onDragOver={this.dragOverHandler}
                onDrop={this.onDropHandler}>
                <div className={classes.title}>Pen2Text</div>
                <div className={classes.editorContainer}>
                    <FileUploader isFileDroping={isDragEnter} />
                    <TextEditor />
                </div>
                <Button className={classes.submitButton}>Submit</Button>
            </div>
        );
    }
}

export default Layout;
