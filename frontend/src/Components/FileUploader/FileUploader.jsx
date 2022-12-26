import React from 'react';
import classes from './FileUploader.scss';
import UploadButton from './UploadButton/UploadButton';
import Button from '@Components/Button/Button';
import DropdownList from '@Components/DropdownList/DropdownList';
import FileView from './FileView/FileView';
import FileList from './FileList/FileList';
import FileItem from './FileList/FileItem/FileItem';
import DragAndDrop from './DragAndDrop/DragAndDrop';
import { FaExchangeAlt } from 'react-icons/fa';
import PenEditor from './PenEditor/PenEditor';
class FileUploader extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            currentFileNo: 0,
            drawnFiles: 0,
            files: [],
            isFileViewerMode: true,
        };
    }

    typeChecker = /^image\/.*|application\/pdf/;

    uploadHandler = event => {
        const uploadedFiles = event.target.files;
        this.updateFiles(uploadedFiles);
    };

    updateFiles = uploadedFiles => {
        const { files } = this.state;
        let { currentFileNo } = this.state;
        currentFileNo = files.length - 1;
        for (let index = 0; index < uploadedFiles.length; index++) {
            if (!this.typeChecker.test(uploadedFiles[index].type)) continue;

            currentFileNo++;
            files.push(uploadedFiles[index]);
        }
        this.props.setFiles(files);
        this.setState({
            files,
            currentFileNo,
        });
    };

    getNextFile = () => {
        const { files } = this.state;
        let { currentFileNo } = this.state;
        currentFileNo++;
        if (currentFileNo > files.length - 1) {
            currentFileNo = 0;
        }
        this.setState({ currentFileNo: currentFileNo });
        return files[currentFileNo];
    };

    getPreviousFile = () => {
        const { files } = this.state;
        let { currentFileNo } = this.state;
        currentFileNo--;
        if (currentFileNo < 0) {
            currentFileNo = files.length - 1;
        }
        this.setState({ currentFileNo: currentFileNo });
        return files[currentFileNo];
    };

    goToSelectedFile = index => {
        this.setState({ currentFileNo: index });
    };

    deleteByIndex = index => {
        const { files } = this.state;
        let { currentFileNo } = this.state;
        files.splice(index, 1);
        if (index <= currentFileNo) {
            currentFileNo--;
            if (currentFileNo < 0) {
                currentFileNo = files.length - 1;
            }
        }
        this.props.setFiles(files);
        this.setState({ currentFileNo: currentFileNo, files: [...files] });
    };

    increaseDrawnFiles = () => {
        let { drawnFiles } = this.state;
        this.setState({ drawnFiles: drawnFiles + 1 });
    };

    changeMode = () => {
        let { isFileViewerMode } = this.state;
        isFileViewerMode = !isFileViewerMode;
        this.setState({ isFileViewerMode });
    };

    setFileViewMode = () => {
        this.setState({ isFileViewerMode: true });
    };

    render() {
        const { isFileDroping } = this.props;
        const { files, currentFileNo, isFileViewerMode, drawnFiles } =
            this.state;
        return (
            <div className={classes.wrapper}>
                <div className={classes.content}>
                    {isFileViewerMode ? (
                        <FileView
                            previosFile={files[currentFileNo - 1]}
                            currentFile={files[currentFileNo]}
                            nextFile={files[currentFileNo + 1]}
                            getNextFile={this.getNextFile}
                            getPreviousFile={this.getPreviousFile}
                            fileCount={files.length}
                        />
                    ) : (
                        <PenEditor
                            updateFiles={this.updateFiles}
                            increaseDrawnFiles={this.increaseDrawnFiles}
                            drawnFiles={drawnFiles}
                        />
                    )}
                    <div className={classes.listButtonWrapper}>
                        <UploadButton uploadHandler={this.uploadHandler} />
                        <FileList>
                            {files.map((element, index) => (
                                <FileItem
                                    id={element.name + index}
                                    key={element.name + index}
                                    index={index}
                                    goToSelectedFile={this.goToSelectedFile}
                                    deleteByIndex={this.deleteByIndex}
                                    setFileViewMode={this.setFileViewMode}>
                                    {element.name}
                                </FileItem>
                            ))}
                        </FileList>
                        <Button
                            className={classes.buttonStyle}
                            onClick={this.changeMode}>
                            <FaExchangeAlt size={15} /> <span>Pen</span>
                        </Button>
                    </div>
                </div>
                {isFileDroping ? (
                    <DragAndDrop onDrop={this.updateFiles} />
                ) : null}
                <DropdownList
                    className={classes.dropdownStyle}
                    items={['English', 'Ukrainian']}
                    setValue={this.props.setLanguage}
                />
            </div>
        );
    }
}

export default FileUploader;
