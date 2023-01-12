import React from 'react';
import classes from './FileView.scss';
import Button from '@Components/Button/Button';

class FileView extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            previousFile: this.props.previousFile,
            currentFile: this.props.currentFile,
            nextFile: this.props.nextFile,
        };
        //this.imgRef = createRef();
    }

    goToNextFile = () => {
        const { currentFile, nextFile } = this.state;
        if (this.props.fileCount < 2) {
            return;
        }

        this.setState({
            previousFile: currentFile,
            currentFile: nextFile,
            nextFile: this.props.getNextFile(),
        });
    };

    goToPreviousFile = () => {
        const { previousFile, currentFile } = this.state;
        if (this.props.fileCount < 2) {
            return;
        }

        this.setState({
            previousFile: this.props.getPreviousFile(),
            currentFile: previousFile,
            nextFile: currentFile,
        });
    };

    imgChecker = /^image\/.*/;

    render() {
        const { currentFile, fileCount } = this.props;
        return (
            <div className={classes.wrapper}>
                {fileCount > 1 ? (
                    <Button
                        className={classes.moveButton}
                        onClick={this.goToPreviousFile}>
                        &#60;
                    </Button>
                ) : null}
                <div
                    className={`${classes.content} ${
                        !this.imgChecker.test(currentFile?.type)
                            ? classes.hideScrollBar
                            : classes.schowScrollBar
                    }`}>
                    {currentFile ? (
                        this.imgChecker.test(currentFile.type) ? (
                            <img
                                ref={this.imgRef}
                                src={URL.createObjectURL(currentFile)}
                                draggable="false"
                            />
                        ) : (
                            <iframe
                                src={URL.createObjectURL(currentFile)}></iframe>
                        )
                    ) : (
                        <span>No files currently selected for upload</span>
                    )}
                </div>
                {fileCount > 1 ? (
                    <Button
                        className={classes.moveButton}
                        onClick={this.goToNextFile}>
                        &#62;
                    </Button>
                ) : null}
            </div>
        );
    }
}

export default FileView;
