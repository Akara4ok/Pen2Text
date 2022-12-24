import React from 'react';
import classes from './FileUploader.scss';
import UploadButton from './UploadButton/UploadButton';
import Button from '@Components/Button/Button';
import DropdownList from '@Components/DropdownList/DropdownList';
import FileView from './FileView/FileView';
import FileList from './FileList/FileList';
import FileItem from './FileList/FileItem/FileItem';
import DragAndDrop from './DragAndDrop/DragAndDrop';
import { FaExchangeAlt } from 'react-icons/fa'
class FileUploader extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            currentFileNo: 0,
            files: [],
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
        this.setState({ currentFileNo: currentFileNo, files: [...files] });
    };

    render() {
        const { isFileDroping } = this.props;
        console.log(isFileDroping);
        const { files, currentFileNo } = this.state;
        return (
            <div className={classes.wrapper}>
                <div className={classes.content}>
                    <FileView
                        previosFile={files[currentFileNo - 1]}
                        currentFile={files[currentFileNo]}
                        nextFile={files[currentFileNo + 1]}
                        getNextFile={this.getNextFile}
                        getPreviousFile={this.getPreviousFile}
                        fileCount={files.length}
                    />
                    <div className={classes.listButtonWrapper}>
                        <UploadButton uploadHandler={this.uploadHandler} />
                        <FileList>
                            {files.map((element, index) => (
                                <FileItem
                                    id={element.name + index}
                                    key={element.name + index}
                                    index={index}
                                    goToSelectedFile={this.goToSelectedFile}
                                    deleteByIndex={this.deleteByIndex}>
                                    {element.name}
                                </FileItem>
                            ))}
                        </FileList>
                        <Button className={classes.buttonStyle}><FaExchangeAlt size={15}/> <span>Pen</span></Button>
                    </div>
                </div>
                {isFileDroping ? (
                    <DragAndDrop onDrop={this.updateFiles} />
                ) : null}
                <DropdownList
                    className={classes.dropdownStyle}
                    items={['English', 'Ukrainian']}
                />
            </div>
        );
    }
}

export default FileUploader;
