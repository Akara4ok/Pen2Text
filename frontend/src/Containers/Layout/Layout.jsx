import React from 'react';
import classes from './Layout.scss';
import FileUploader from '@Components/FileUploader/FileUploader';
import TextEditor from '@Components/TextEditor/TextEditor';
import Button from '@Components/Button/Button';

class Layout extends React.Component {
    constructor(props) {
        super(props);
    }
    render() {
        return (
            <div className={classes.wrapper}>
                <div className={classes.title}>Pen2Text</div>
                <div className={classes.editorContainer}>
                    <FileUploader />
                    <TextEditor />
                </div>
                <Button>Submit</Button>
            </div>
        );
    }
}

export default Layout;
